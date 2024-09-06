import math
import random
from enum import Enum


random.seed(10)


class EventType(Enum):
    ARRIVE = 0
    EXIT_QUEUE_1 = 1
    EXIT_QUEUE_2 = 2


ARRIVAL_RATE = 3
SERVICE_TIME_STAGE_1 = 5
SERVICE_TIME_STAGE_2 = 7
SIM_TIME = 50

queue_stage_2 = []

current_time = 0

server_stage_1_busy = False
server_stage_2_busy = False

total_jobs = 0
total_time_in_system = 0


class Event:
    def __init__(self, time, type, job_id):
        self.time = time
        self.type = type
        self.job_id = job_id


class EventStackNode:
    def __init__(self, event):
        self.event = event
        self.next = None
        self.prev = None


class EventStack:
    def __init__(self):
        self.head = None
        self.tail = None

    def is_empty(self):
        return self.head == None

    def insert(self, event):
        node = EventStackNode(event)

        if self.is_empty():
            self.head = node
            self.tail = node
            return

        if node.event.time <= self.head.event.time:
            node.next = self.head
            self.head.prev = node
            self.head = node
            return

        if node.event.time >= self.tail.event.time:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
            return

        current = self.head
        while current and current.event.time < node.event.time:
            current = current.next

        node.next = current
        node.prev = current.prev

        if current.prev:
            current.prev.next = node
        current.prev = node

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from an empty priority queue")

        # Pop the head (lowest time stamp event)
        event = self.head.event
        self.head = self.head.next

        if self.head:
            self.head.prev = None
        else:
            self.tail = None

        return event


# Priority queue to manage events
event_stack = EventStack()


def arrival():
    """Handles job arrival."""
    global total_jobs, server_stage_1_busy

    total_jobs += 1
    job_id = total_jobs
    print(f"Job {job_id} arrives at time {current_time:.2f}")

    # Schedule the next arrival (Poisson process)
    inter_arrival_time = random.expovariate(ARRIVAL_RATE)
    event = Event(current_time + inter_arrival_time, EventType.ARRIVE, None)
    event_stack.insert(event)

    # If server for stage 1 is free, start processing the job
    if not server_stage_1_busy:
        process_stage_1(job_id)
    else:
        # If the server is busy, the job has to wait in stage 1 queue
        queue_stage_2.append(job_id)


def process_stage_1(job_id):
    """Processes a job in stage 1."""
    global server_stage_1_busy
    server_stage_1_busy = True

    print(f"Job {job_id} starts service at stage 1 at {current_time:.2f}")

    # Schedule the completion of stage 1
    service_time_1 = random.expovariate(1.0 / SERVICE_TIME_STAGE_1)
    event = Event(current_time + service_time_1, EventType.EXIT_QUEUE_1, job_id)
    event_stack.insert(event)


def complete_stage_1(job_id):
    """Handles the completion of stage 1 for a job."""
    global server_stage_1_busy, server_stage_2_busy

    print(f"Job {job_id} completes stage 1 at {current_time:.2f}")
    server_stage_1_busy = False

    # If server for stage 2 is free, move the job to stage 2
    if not server_stage_2_busy:
        process_stage_2(job_id)
    else:
        # Queue the job for stage 2
        queue_stage_2.append(job_id)

    # If there are more jobs waiting in queue for stage 1, start the next job
    if queue_stage_2:
        next_job_id = queue_stage_2.pop(0)
        process_stage_1(next_job_id)


def process_stage_2(job_id):
    """Processes a job in stage 2."""
    global server_stage_2_busy
    server_stage_2_busy = True

    print(f"Job {job_id} starts service at stage 2 at {current_time:.2f}")

    # Schedule the completion of stage 2
    service_time_2 = random.expovariate(1.0 / SERVICE_TIME_STAGE_2)
    event = Event(current_time + service_time_2, EventType.EXIT_QUEUE_2, job_id)
    event_stack.insert(event)


def complete_stage_2(job_id):
    """Handles the completion of stage 2 for a job."""
    global server_stage_2_busy, total_time_in_system

    print(f"Job {job_id} completes stage 2 at {current_time:.2f}")
    server_stage_2_busy = False

    # Track total time in the system for the job
    total_time_in_system += current_time

    # If there are more jobs waiting in queue for stage 2, start the next job
    if queue_stage_2:
        next_job_id = queue_stage_2.pop(0)
        process_stage_2(next_job_id)


def main():
    global current_time

    # Schedule the first job arrival to kick off the simulation
    event = Event(0, EventType.ARRIVE, None)
    event_stack.insert(event)

    # Run the simulation loop
    while event_stack and current_time < SIM_TIME:
        # Get the next event in the event queue (min-heap)
        event = event_stack.pop()
        current_time, event_type, job_id = event.time, event.type, event.job_id

        # Handle the event based on its type
        if event_type == EventType.ARRIVE:
            arrival()
        elif event_type == EventType.EXIT_QUEUE_1:
            complete_stage_1(job_id)
        elif event_type == EventType.EXIT_QUEUE_2:
            complete_stage_2(job_id)
        else:
            assert False, "UNREACHABLE"

    # Print final statistics
    average_time_in_system = total_time_in_system / total_jobs if total_jobs > 0 else 0
    print(f"\nTotal jobs processed: {total_jobs}")
    print(f"Average time in system: {average_time_in_system:.2f} units of time")


if __name__ == "__main__":
    main()
