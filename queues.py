from collections import defaultdict
import math
import random
from enum import Enum


random.seed(10)


def exponential_random(mean):
    U = random.random()
    return -math.log(1 - U) / mean


class EventType(Enum):
    ARRIVAL = 0
    COMPLETE_STAGE_1 = 1
    COMPLETE_STAGE_2 = 2


ARRIVAL_RATE = 1
SERVICE_TIME_STAGE_1 = 3
SERVICE_TIME_STAGE_2 = 4
SIM_TIME = 10000

q1 = []
q2 = []

current_time = 0

server_1_busy = False
server_2_busy = False

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

        event = self.head.event
        self.head = self.head.next

        if self.head:
            self.head.prev = None
        else:
            self.tail = None

        return event


event_stack = EventStack()


def arrival():
    """Handles job arrival."""
    global total_jobs, server_1_busy

    total_jobs += 1
    job_id = total_jobs
    print(f"Job {job_id} arrives at time {current_time:.2f}")

    # Schedule the next arrival (Poisson process)
    inter_arrival_time = exponential_random(ARRIVAL_RATE)
    event = Event(current_time + inter_arrival_time, EventType.ARRIVAL, None)
    event_stack.insert(event)

    # If server for stage 1 is free, start processing the job
    if not server_1_busy:
        process_stage_1(job_id)
    else:
        # If the server is busy, the job has to wait in stage 1 queue
        q1.append(job_id)


def process_stage_1(job_id):
    """Processes a job in stage 1."""
    global server_1_busy
    server_1_busy = True

    print(f"Job {job_id} starts service at stage 1 at {current_time:.2f}")

    # Schedule the completion of stage 1
    service_time_1 = exponential_random(SERVICE_TIME_STAGE_1)
    event = Event(current_time + service_time_1, EventType.COMPLETE_STAGE_1, job_id)
    event_stack.insert(event)


def complete_stage_1(job_id):
    """Handles the completion of stage 1 for a job."""
    global server_1_busy, server_2_busy

    print(f"Job {job_id} completes stage 1 at {current_time:.2f}")
    server_1_busy = False

    # If server for stage 2 is free, move the job to stage 2
    if not server_2_busy:
        process_stage_2(job_id)
    else:
        # Queue the job for stage 2
        q2.append(job_id)

    # If there are more jobs waiting in queue for stage 1, start the next job
    if len(q1) > 0:
        next_job_id = q1.pop(0)
        process_stage_1(next_job_id)


def process_stage_2(job_id):
    """Processes a job in stage 2."""
    global server_2_busy
    server_2_busy = True

    print(f"Job {job_id} starts service at stage 2 at {current_time:.2f}")

    # Schedule the completion of stage 2
    service_time_2 = exponential_random(SERVICE_TIME_STAGE_2)
    event = Event(current_time + service_time_2, EventType.COMPLETE_STAGE_2, job_id)
    event_stack.insert(event)


def complete_stage_2(job_id):
    """Handles the completion of stage 2 for a job."""
    global server_2_busy, total_time_in_system

    print(f"Job {job_id} completes stage 2 at {current_time:.2f}")
    server_2_busy = False

    # Track total time in the system for the job
    total_time_in_system += current_time

    # If there are more jobs waiting in queue for stage 2, start the next job
    if len(q2) > 0:
        next_job_id = q2.pop(0)
        process_stage_2(next_job_id)


def main():
    global current_time

    event = Event(0, EventType.ARRIVAL, None)
    event_stack.insert(event)

    start_time = 0
    queue_1_len = 0
    queue_2_len = 0

    q1_map = defaultdict(lambda: 0)
    q2_map = defaultdict(lambda: 0)
    qq_map = defaultdict(lambda: 0)
    while not event_stack.is_empty() and current_time < SIM_TIME:
        event = event_stack.pop()
        current_time, event_type, job_id = event.time, event.type, event.job_id
        elapsed = current_time - start_time
        q1_map[queue_1_len] += elapsed
        q2_map[queue_2_len] += elapsed
        qq_map[queue_1_len + queue_2_len] += elapsed

        queue_1_len = len(q1)
        queue_2_len = len(q2)
        start_time = current_time

        if event_type == EventType.ARRIVAL:
            arrival()
        elif event_type == EventType.COMPLETE_STAGE_1:
            complete_stage_1(job_id)
        elif event_type == EventType.COMPLETE_STAGE_2:
            complete_stage_2(job_id)
        else:
            assert False, "UNREACHABLE"

    for length, freq in freq_to_prob(q1_map).items():
        print(length, freq)

    print()

    for length, freq in freq_to_prob(q2_map).items():
        print(length, freq)

    print()

    for length, freq in freq_to_prob(qq_map).items():
        print(length, freq)

    average_time_in_system = total_time_in_system / total_jobs if total_jobs > 0 else 0
    print(f"\nTotal jobs processed: {total_jobs}")
    print(f"Average time in system: {average_time_in_system:.2f} units of time")


def freq_to_prob(freqs):
    result = {}
    total_freq = 0
    for _, freq in freqs.items():
        total_freq += freq

    for lenght, freq in freqs.items():
        result[lenght] = freqs[lenght] / total_freq

    return result


if __name__ == "__main__":
    main()
