"""Tests for the safety rules. Run: python -m unittest discover tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from extract import extract


class TestExtractionSafety(unittest.TestCase):
    def test_missing_information_is_null_not_invented(self):
        note = (
            "SYNTHETIC DISCHARGE NOTE SYN-TEST. Diagnosis: heart failure. "
            "Medications: none documented. Follow-up: not scheduled. "
            "Red flags: none documented. Care team contact: not documented."
        )
        result = extract(note)
        self.assertEqual(result["medications"], [])
        self.assertIsNone(result["follow_up_days"])
        self.assertIsNone(result["follow_up_clinic"])
        self.assertEqual(result["red_flags"], [])
        self.assertIsNone(result["contact"])

    def test_medication_is_extracted_verbatim(self):
        note = (
            "SYNTHETIC DISCHARGE NOTE SYN-TEST. Medications: lisinopril 10 mg daily. "
            "Follow-up: in 7 days with cardiology clinic. Red flags: chest pain. "
            "Care team contact: 555-0199."
        )
        result = extract(note)
        self.assertEqual(
            result["medications"],
            [{"name": "lisinopril", "dose": "10 mg", "frequency": "daily"}],
        )
        self.assertEqual(result["follow_up_days"], 7)
        self.assertEqual(result["contact"], "555-0199")


if __name__ == "__main__":
    unittest.main()
