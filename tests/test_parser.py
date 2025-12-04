"""Tests for the parser module."""

import pytest
import numpy as np
from minter.parser import (
    calculate_overall_dn0,
    CANONICAL_NET_TYPES,
    DN0Result,
)


class TestCalculateOverallDn0:
    """Tests for calculate_overall_dn0 function."""

    def test_returns_dn0_result(self):
        """Test that function returns a DN0Result namedtuple."""
        # Note: This will fail without actual ITN params file
        # In real tests, you'd mock the file loading
        pass

    def test_canonical_name_mapping(self):
        """Test that canonical name mapping is correct."""
        assert CANONICAL_NET_TYPES["py_only"] == "pyrethroid_only"
        assert CANONICAL_NET_TYPES["py_pbo"] == "pyrethroid_pbo"
        assert CANONICAL_NET_TYPES["py_pyrrole"] == "pyrethroid_pyrrole"
        assert CANONICAL_NET_TYPES["py_ppf"] == "pyrethroid_ppf"

    def test_empty_usage_raises_error(self):
        """Test that empty usage values raises an error."""
        with pytest.raises(ValueError, match="must supply at least one"):
            calculate_overall_dn0(resistance_level=0.5)

    def test_unknown_net_type_raises_error(self):
        """Test that unknown net types raise an error."""
        with pytest.raises(ValueError, match="Unknown net type"):
            calculate_overall_dn0(
                resistance_level=0.5,
                unknown_net=0.5,
            )


class TestDN0Result:
    """Tests for DN0Result namedtuple."""

    def test_dn0_result_fields(self):
        """Test DN0Result has correct fields."""
        result = DN0Result(dn0=0.5, itn_use=0.8)
        assert result.dn0 == 0.5
        assert result.itn_use == 0.8

    def test_dn0_result_immutable(self):
        """Test DN0Result is immutable."""
        result = DN0Result(dn0=0.5, itn_use=0.8)
        with pytest.raises(AttributeError):
            result.dn0 = 0.6
