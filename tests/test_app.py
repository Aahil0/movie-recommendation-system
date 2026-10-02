from pathlib import Path
import unittest
from streamlit.testing.v1 import AppTest


class StreamlitTests(unittest.TestCase):
    def test_search_modes_counts_and_persistence(self):
        app=AppTest.from_file(str(Path(__file__).resolve().parents[1]/"app.py"),default_timeout=60).run()
        self.assertEqual(len(app.exception),0)
        app.selectbox[0].select("Matrix, The (1999)")
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        result=app.session_state["film_results"]
        self.assertEqual(len(result),10)
        self.assertNotIn("Matrix, The (1999)",set(result.title))
        app.run()
        self.assertEqual(len(app.session_state["film_results"]),10)
        app.slider[0].set_value(20)
        app.selectbox[1].select("Genre matches")
        app.button[0].click().run()
        self.assertEqual(len(app.exception),0)
        self.assertEqual(len(app.session_state["film_results"]),20)
        self.assertEqual(app.session_state["film_mode"],"Genre matches")
        self.assertTrue(any("film-grid" in element.value and "<article" in element.value for element in app.markdown))


if __name__=="__main__":unittest.main()
