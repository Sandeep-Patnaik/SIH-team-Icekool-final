import pytest
from sql_guard import validate_sql, UnsafeSQLError

def test_valid_select_statements():
    assert validate_sql("SELECT * FROM measurements;") == "SELECT * FROM measurements;"
    assert validate_sql("SELECT temperature, salinity FROM measurements JOIN profiles ON measurements.profile_id = profiles.profile_id;")

def test_drop_rejection():
    with pytest.raises(UnsafeSQLError):
        validate_sql("DROP TABLE floats;")

def test_delete_rejection():
    with pytest.raises(UnsafeSQLError):
        validate_sql("DELETE FROM profiles WHERE cycle_number = 1;")

def test_update_rejection():
    with pytest.raises(UnsafeSQLError):
        validate_sql("UPDATE measurements SET temperature = 10.0;")

def test_insert_rejection():
    with pytest.raises(UnsafeSQLError):
        validate_sql("INSERT INTO reports (report_id) VALUES ('r1');")

def test_union_injection_rejection():
    with pytest.raises(UnsafeSQLError):
        validate_sql("SELECT temperature FROM measurements UNION SELECT password FROM users;")

def test_stacked_statements_rejection():
    with pytest.raises(UnsafeSQLError):
        validate_sql("SELECT * FROM measurements; DROP TABLE profiles;")