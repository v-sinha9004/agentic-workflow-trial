import unittest
from agent.dag import WorkflowDAG, DAGCycleError, create_flight_booking_dag


class TestWorkflowDAG(unittest.TestCase):
    def test_default_flight_dag_structure(self):
        dag = create_flight_booking_dag()
        self.assertIn("search_flights", dag.nodes)
        self.assertIn("get_baggage_policy", dag.nodes)
        self.assertIn("calculator", dag.nodes)
        self.assertIn("book_flight", dag.nodes)

        # Initial state
        can_search, unmet = dag.can_execute("search_flights")
        self.assertTrue(can_search)
        self.assertEqual(unmet, [])

        # book_flight should be blocked initially
        can_book, unmet_book = dag.can_execute("book_flight")
        self.assertFalse(can_book)
        self.assertIn("search_flights", unmet_book)
        self.assertIn("calculator", unmet_book)

    def test_dag_prerequisite_progression(self):
        dag = create_flight_booking_dag()

        # Step 1: Complete search_flights
        dag.mark_completed("search_flights")
        can_baggage, _ = dag.can_execute("get_baggage_policy")
        can_calc, _ = dag.can_execute("calculator")
        can_book, unmet_book = dag.can_execute("book_flight")

        self.assertTrue(can_baggage)
        self.assertTrue(can_calc)
        self.assertFalse(can_book)
        self.assertEqual(unmet_book, ["calculator"])

        # Step 2: Complete calculator
        dag.mark_completed("calculator")
        can_book, unmet_book = dag.can_execute("book_flight")
        self.assertTrue(can_book)
        self.assertEqual(unmet_book, [])

    def test_dag_cycle_detection(self):
        dag = WorkflowDAG()
        dag.add_step("stepA", dependencies=[])
        dag.add_step("stepB", dependencies=["stepA"])

        # Introducing a cycle: stepA depending on stepB
        with self.assertRaises(DAGCycleError):
            dag.add_step("stepA", dependencies=["stepB"])

    def test_unconstrained_step_always_executable(self):
        dag = create_flight_booking_dag()
        # A utility step not in DAG (e.g. get_city_weather)
        can_weather, unmet = dag.can_execute("get_city_weather")
        self.assertTrue(can_weather)
        self.assertEqual(unmet, [])

    def test_dag_reset(self):
        dag = create_flight_booking_dag()
        dag.mark_completed("search_flights")
        dag.mark_completed("calculator")
        can_book, _ = dag.can_execute("book_flight")
        self.assertTrue(can_book)

        dag.reset()
        can_book, unmet = dag.can_execute("book_flight")
        self.assertFalse(can_book)
        self.assertEqual(len(unmet), 2)

    def test_dag_render_ascii(self):
        dag = create_flight_booking_dag()
        rendered = dag.render_ascii()
        self.assertIn("Workflow DAG Status", rendered)
        self.assertIn("search_flights", rendered)
        self.assertIn("book_flight", rendered)


if __name__ == "__main__":
    unittest.main()
