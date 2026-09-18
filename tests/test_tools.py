import unittest
import json
from agent.tools import registry, ToolRegistry, tool
from agent.config import load_config


class TestToolRegistry(unittest.TestCase):
    def test_registered_tools_exist(self):
        self.assertIsNotNone(registry.get("search_flights"))
        self.assertIsNotNone(registry.get("get_baggage_policy"))
        self.assertIsNotNone(registry.get("calculator"))
        self.assertIsNotNone(registry.get("get_city_weather"))
        self.assertIsNotNone(registry.get("book_flight"))

    def test_schema_structure(self):
        schemas = registry.get_schemas()
        self.assertGreaterEqual(len(schemas), 5)
        for s in schemas:
            self.assertEqual(s["type"], "function")
            self.assertIn("name", s["function"])
            self.assertIn("description", s["function"])
            self.assertIn("parameters", s["function"])
            params = s["function"]["parameters"]
            self.assertEqual(params["type"], "object")

    def test_calculator_safe_eval(self):
        calc = registry.get("calculator")
        self.assertEqual(calc.execute(expression="(100 * 2) + 50"), "250")
        self.assertEqual(calc.execute(expression="$450 + $60"), "510")

    def test_calculator_unsafe_rejection(self):
        calc = registry.get("calculator")
        err = calc.execute(expression="__import__('os').system('ls')")
        self.assertTrue("Error" in err)

    def test_search_flights(self):
        search = registry.get("search_flights")
        res = search.execute(origin="JFK", destination="LHR", date="2026-09-25")
        data = json.loads(res)
        self.assertIn("flights_found", data)
        self.assertGreaterEqual(len(data["flights_found"]), 1)

    def test_book_flight_execution(self):
        book_tool = registry.get("book_flight")
        self.assertIsNotNone(book_tool)
        res = book_tool.execute(flight_number="BA 178", passenger_name="Alice Smith", date="2026-09-25")
        data = json.loads(res)
        self.assertEqual(data["status"], "CONFIRMED")
        self.assertEqual(data["flight_number"], "BA 178")
        self.assertEqual(data["passenger_name"], "Alice Smith")
        self.assertTrue(data["booking_reference"].startswith("BK-BA178-"))

    def test_requires_confirmation_flag(self):
        book_tool = registry.get("book_flight")
        search_tool = registry.get("search_flights")
        self.assertTrue(book_tool.requires_confirmation)
        self.assertFalse(search_tool.requires_confirmation)

    def test_custom_tool_registration(self):
        custom_registry = ToolRegistry()

        @tool(name="sample_tool", description="A test tool", requires_confirmation=True)
        def sample(a: int, b: str = "default") -> str:
            """Sample docstring."""
            return f"{a}:{b}"

        custom_registry.register(sample)
        tool_obj = custom_registry.get("sample_tool")
        self.assertIsNotNone(tool_obj)
        self.assertTrue(tool_obj.requires_confirmation)
        schemas = custom_registry.get_schemas()
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0]["function"]["name"], "sample_tool")


if __name__ == "__main__":
    unittest.main()

