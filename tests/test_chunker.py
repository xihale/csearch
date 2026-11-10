import unittest

from csearch.chunker import chunk_code


class ChunkBoundaryTests(unittest.TestCase):
    def test_overlap_retains_source_line_ranges(self):
        source = "ab\ncd\nef\ngh\n"
        chunks = chunk_code(source, max_chars=6, overlap_chars=3)
        self.assertEqual(chunks, [
            (0, 2, "ab\ncd\n"),
            (1, 3, "cd\nef\n"),
            (2, 4, "ef\ngh\n"),
        ])

    def test_long_line_resets_buffer_before_following_lines(self):
        chunks = chunk_code("ab\n123456789\ncd\nef\n", max_chars=6, overlap_chars=0)
        self.assertEqual(chunks, [
            (0, 1, "ab\n"),
            (1, 2, "123456"),
            (1, 2, "789\n"),
            (2, 4, "cd\nef\n"),
        ])


if __name__ == "__main__":
    unittest.main()
