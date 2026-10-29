from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
import pytest

from lastfm_project.data_loading import (
    ArchiveIntegrityError,
    DataValidationError,
    load_artists,
    load_hetrec,
    load_user_artists,
    load_user_friends,
)


ARCHIVE = Path("data/raw/hetrec2011-lastfm-2k.zip")


def write_fixture(path: Path, *, artists: str, interactions: str, friends: str) -> Path:
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("artists.dat", artists.encode("utf-8"))
        archive.writestr("user_artists.dat", interactions.encode("utf-8"))
        archive.writestr("user_friends.dat", friends.encode("utf-8"))
    return path


@pytest.fixture
def valid_zip(tmp_path: Path) -> Path:
    return write_fixture(
        tmp_path / "fixture.zip",
        artists="id\tname\turl\tpictureURL\n2\tBeyoncé\thttps://b\t\n1\tA\thttps://a\thttps://pic\n",
        interactions="userID\tartistID\tweight\n2\t2\t4\n1\t1\t3\n",
        friends="userID\tfriendID\n2\t1\n1\t2\n",
    )


def test_valid_fixture_schemas_ordering_nullable_and_mappings(valid_zip: Path) -> None:
    data = load_hetrec(valid_zip)
    assert list(data.artists.columns) == ["artist_id", "name", "url", "picture_url"]
    assert data.artists["artist_id"].dtype == "int64"
    assert str(data.artists["name"].dtype) == "string"
    assert pd.isna(data.artists.loc[1, "picture_url"])
    assert data.interactions.to_dict("records") == [
        {"user_id": 1, "artist_id": 1, "weight": 3},
        {"user_id": 2, "artist_id": 2, "weight": 4},
    ]
    assert data.friendships.directed.to_dict("records") == [
        {"user_id": 1, "friend_id": 2},
        {"user_id": 2, "friend_id": 1},
    ]
    assert data.friendships.undirected.to_dict("records") == [{"user_low_id": 1, "user_high_id": 2}]
    assert data.user_index.to_dict("records") == [{"user_id": 1, "index": 0}, {"user_id": 2, "index": 1}]
    assert data.artist_index.to_dict("records") == [{"artist_id": 1, "index": 0}, {"artist_id": 2, "index": 1}]


@pytest.mark.parametrize(
    ("member", "content", "message"),
    [
        ("artists.dat", "bad\theader\tx\ty\n1\tA\tu\tp\n", "header"),
        ("user_artists.dat", "userID\tartistID\tweight\n1\t1\n", "fields"),
        ("artists.dat", "id\tname\turl\tpictureURL\n1\t\tu\tp\n", "null field"),
    ],
)
def test_invalid_headers_rows_and_nulls(tmp_path: Path, member: str, content: str, message: str) -> None:
    contents = {
        "artists.dat": "id\tname\turl\tpictureURL\n1\tA\tu\tp\n",
        "user_artists.dat": "userID\tartistID\tweight\n1\t1\t2\n",
        "user_friends.dat": "userID\tfriendID\n1\t2\n2\t1\n",
    }
    contents[member] = content
    path = write_fixture(
        tmp_path / "bad.zip",
        artists=contents["artists.dat"],
        interactions=contents["user_artists.dat"],
        friends=contents["user_friends.dat"],
    )
    with pytest.raises(DataValidationError, match=message):
        load_hetrec(path)


@pytest.mark.parametrize(
    ("interactions", "friends", "message"),
    [
        ("userID\tartistID\tweight\n1\t1\t0\n", "userID\tfriendID\n1\t2\n2\t1\n", "positive"),
        ("userID\tartistID\tweight\n1\t1\t2\n1\t1\t3\n", "userID\tfriendID\n1\t2\n2\t1\n", "duplicate"),
        ("userID\tartistID\tweight\n1\t9\t2\n", "userID\tfriendID\n1\t2\n2\t1\n", "foreign keys"),
        (
            "userID\tartistID\tweight\n1\t1\t2\n2\t1\t2\n",
            "userID\tfriendID\n1\t2\n",
            "reciprocal",
        ),
        ("userID\tartistID\tweight\n1\t1\t2\n", "userID\tfriendID\n1\t1\n", "self-loop"),
        ("userID\tartistID\tweight\n1\t1\t2\n", "userID\tfriendID\n1\t2\n2\t1\n", "endpoint users"),
    ],
)
def test_cross_table_and_key_invariants(tmp_path: Path, interactions: str, friends: str, message: str) -> None:
    path = write_fixture(
        tmp_path / "invalid.zip",
        artists="id\tname\turl\tpictureURL\n1\tA\tu\tp\n",
        interactions=interactions,
        friends=friends,
    )
    with pytest.raises(DataValidationError, match=message):
        load_hetrec(path)


@pytest.mark.parametrize(
    ("artists", "interactions", "friends", "message"),
    [
        (
            "id\tname\turl\tpictureURL\n1\tA\tu\tp\n1\tB\tv\tp\n",
            "userID\tartistID\tweight\n1\t1\t2\n",
            "userID\tfriendID\n",
            "duplicate artist_id",
        ),
        (
            "id\tname\turl\tpictureURL\n0\tA\tu\tp\n",
            "userID\tartistID\tweight\n1\t0\t2\n",
            "userID\tfriendID\n",
            "must be positive",
        ),
    ],
)
def test_duplicate_and_non_positive_ids_are_rejected(
    tmp_path: Path, artists: str, interactions: str, friends: str, message: str
) -> None:
    path = write_fixture(tmp_path / "bad_ids.zip", artists=artists, interactions=interactions, friends=friends)
    with pytest.raises(DataValidationError, match=message):
        load_hetrec(path)


@pytest.mark.parametrize(
    ("member", "content", "field", "message"),
    [
        (
            "user_artists.dat",
            "userID\tartistID\tweight\n1\t9223372036854775808\t2\n",
            "artistID",
            "outside signed int64 range",
        ),
        (
            "user_friends.dat",
            "userID\tfriendID\n-9223372036854775809\t1\n1\t-9223372036854775809\n",
            "userID",
            "outside signed int64 range",
        ),
        (
            "user_artists.dat",
            "userID\tartistID\tweight\n1x\t1\t2\n",
            "userID",
            "not an integer",
        ),
    ],
)
def test_integer_fields_are_validated_before_pandas_conversion(
    tmp_path: Path, member: str, content: str, field: str, message: str
) -> None:
    contents = {
        "artists.dat": "id\tname\turl\tpictureURL\n1\tA\tu\tp\n",
        "user_artists.dat": "userID\tartistID\tweight\n1\t1\t2\n",
        "user_friends.dat": "userID\tfriendID\n1\t2\n2\t1\n",
    }
    contents[member] = content
    path = write_fixture(
        tmp_path / "bad_integer.zip",
        artists=contents["artists.dat"],
        interactions=contents["user_artists.dat"],
        friends=contents["user_friends.dat"],
    )
    with pytest.raises(DataValidationError, match=rf"{member}: row 2 field '{field}'.*{message}"):
        load_hetrec(path)


def test_duplicate_zip_member_names_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("artists.dat", "id\tname\turl\tpictureURL\n1\tA\tu\tp\n")
        archive.writestr("artists.dat", "id\tname\turl\tpictureURL\n1\tA\tu\tp\n")
        archive.writestr("user_artists.dat", "userID\tartistID\tweight\n1\t1\t2\n")
        archive.writestr("user_friends.dat", "userID\tfriendID\n1\t1\n")
    with pytest.raises(ArchiveIntegrityError, match="duplicate member names"):
        load_hetrec(path)


def test_missing_required_zip_member_is_rejected(valid_zip: Path, tmp_path: Path) -> None:
    path = tmp_path / "missing_required_member.zip"
    with ZipFile(valid_zip) as source, ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for name in ("artists.dat", "user_artists.dat"):
            archive.writestr(name, source.read(name))
    with pytest.raises(ArchiveIntegrityError, match="missing required members"):
        load_hetrec(path)


def test_invalid_utf8_in_required_source_table_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "invalid_utf8.zip"
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        archive.writestr("artists.dat", b"id\tname\turl\tpictureURL\n1\t\xff\tu\tp\n")
        archive.writestr("user_artists.dat", "userID\tartistID\tweight\n1\t1\t2\n")
        archive.writestr("user_friends.dat", "userID\tfriendID\n1\t1\n")
    with pytest.raises(DataValidationError, match="artists.dat: invalid UTF-8"):
        load_hetrec(path)


def test_missing_final_newline_in_required_source_table_is_rejected(tmp_path: Path) -> None:
    path = write_fixture(
        tmp_path / "missing_newline.zip",
        artists="id\tname\turl\tpictureURL\n1\tA\tu\tp",
        interactions="userID\tartistID\tweight\n1\t1\t2\n",
        friends="userID\tfriendID\n1\t1\n",
    )
    with pytest.raises(DataValidationError, match="artists.dat: missing final newline"):
        load_hetrec(path)


def test_crc_is_checked(valid_zip: Path, tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.zip"
    with ZipFile(valid_zip) as source, ZipFile(corrupt, "w") as destination:
        for name in ("artists.dat", "user_artists.dat", "user_friends.dat"):
            destination.writestr(name, source.read(name), compress_type=0)
    with ZipFile(corrupt) as archive:
        info = archive.getinfo("artists.dat")
        payload_offset = info.header_offset + 30 + len(info.filename.encode("utf-8")) + len(info.extra)
    raw = bytearray(corrupt.read_bytes())
    raw[payload_offset] ^= 0x01
    corrupt.write_bytes(raw)
    with pytest.raises(ArchiveIntegrityError, match="CRC"):
        load_artists(corrupt)


def test_original_archive_audit_counts_and_integrity() -> None:
    data = load_hetrec(ARCHIVE)
    assert len(data.interactions) == 92_834
    assert data.interactions["user_id"].nunique() == 1_892
    assert data.interactions["artist_id"].nunique() == 17_632
    assert int(data.interactions["weight"].sum()) == 69_183_975
    assert len(data.artists) == 17_632
    assert data.artists["picture_url"].isna().sum() == 444
    assert len(data.friendships.directed) == 25_434
    assert len(data.friendships.undirected) == 12_717
    assert len(data.user_index) == 1_892
    assert len(data.artist_index) == 17_632
