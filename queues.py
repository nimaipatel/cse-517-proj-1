# Copyright (C) 2024 Patel, Nimai <nimai.m.patel@gmail.com>
# Author: Patel, Nimai <nimai.m.patel@gmail.com>

from __future__ import annotations
from collections import defaultdict
import math
import random
from enum import Enum
from typing import List, NewType, Optional


random.seed(10)

job_id_t = NewType('job_id_t', int)


class EventType(Enum):
    ARRIVAL = 0
    COMPLETE_STAGE_1 = 1
    COMPLETE_STAGE_2 = 2


ARRIVAL_RATE = 1
SERVICE_TIME_STAGE_1 = 3
SERVICE_TIME_STAGE_2 = 4
SIM_TIME = 10000

q1: List[job_id_t] = []
q2: List[job_id_t] = []

curr_time = 0

server_1_busy = False
server_2_busy = False

total_jobs = 0
comp_jobs = 0
total_sojourn_time: float = 0
arrival_times: dict[job_id_t, float] = {}


def exponential_random(mean: float):
    U = random.random()
    return -math.log(1 - U) / mean


class Event:
    def __init__(self, time: float, type: EventType, job_id: job_id_t):
        self.time = time
        self.type = type
        self.job_id = job_id


class EventStackNode:
    def __init__(self, event: Event):
        self.event = event
        self.next: Optional[EventStackNode] = None
        self.prev: Optional[EventStackNode] = None


class EventStack:
    def __init__(self):
        self.head = None
        self.tail = None

    def is_empty(self):
        return self.head == None

    def insert(self, event: Event):
        node = EventStackNode(event)

        if self.is_empty():
            self.head = node
            self.tail = node
            return

        assert (
            self.head != None
        ), "head cannot be None, we checked that the queue is not empty"
        if node.event.time <= self.head.event.time:
            node.next = self.head
            self.head.prev = node
            self.head = node
            return

        assert (
            self.tail != None
        ), "tail cannot be None, we checked that the queue is not empty"
        if node.event.time >= self.tail.event.time:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
            return

        curr = self.head
        while curr and curr.event.time < node.event.time:
            curr = curr.next

        assert (
            curr != None
        ), "curr can't be None, insertion at end of the stack is handled in O(1) already"
        node.next = curr
        node.prev = curr.prev

        if curr.prev:
            curr.prev.next = node
        curr.prev = node

    def pop(self):
        assert self.head != None
        event = self.head.event
        self.head = self.head.next

        if self.head:
            self.head.prev = None
        else:
            self.tail = None

        return event


es = EventStack()


def arrival():
    """Handles job arrival."""
    global total_jobs, server_1_busy

    total_jobs += 1
    job_id = job_id_t(total_jobs)
    print(f"Job {job_id} arrives at time {curr_time:.2f}")
    arrival_times[job_id] = curr_time

    # Schedule the next arrival (Poisson process)
    inter_arrival_time = exponential_random(ARRIVAL_RATE)
    event = Event(curr_time + inter_arrival_time, EventType.ARRIVAL, job_id)
    es.insert(event)

    # If server for stage 1 is free, start processing the job
    if server_1_busy:
        q1.append(job_id)
    else:
        process_stage_1(job_id)


def process_stage_1(job_id: job_id_t):
    """Processes a job in stage 1."""
    global server_1_busy
    server_1_busy = True

    print(f"Job {job_id} starts service at stage 1 at {curr_time:.2f}")

    # Schedule the completion of stage 1
    service_time_1 = exponential_random(SERVICE_TIME_STAGE_1)
    event = Event(curr_time + service_time_1, EventType.COMPLETE_STAGE_1, job_id)
    es.insert(event)


def complete_stage_1(job_id: job_id_t):
    """Handles the completion of stage 1 for a job."""
    global server_1_busy, server_2_busy

    print(f"Job {job_id} completes stage 1 at {curr_time:.2f}")
    server_1_busy = False

    # If server for stage 2 is free, move the job to stage 2
    if server_2_busy:
        q2.append(job_id)
    else:
        process_stage_2(job_id)

    # If there are more jobs waiting in queue for stage 1, start the next job
    if len(q1) > 0:
        next_job_id = q1.pop(0)
        process_stage_1(next_job_id)


def process_stage_2(job_id: job_id_t):
    """Processes a job in stage 2."""
    global server_2_busy
    server_2_busy = True

    print(f"Job {job_id} starts service at stage 2 at {curr_time:.2f}")

    # Schedule the completion of stage 2
    service_time_2 = exponential_random(SERVICE_TIME_STAGE_2)
    event = Event(curr_time + service_time_2, EventType.COMPLETE_STAGE_2, job_id)
    es.insert(event)


def complete_stage_2(job_id: job_id_t):
    """Handles the completion of stage 2 for a job."""
    global server_2_busy, total_sojourn_time, comp_jobs

    print(f"Job {job_id} completes stage 2 at {curr_time:.2f}")
    server_2_busy = False

    # Track total time in the system for the job
    total_sojourn_time += curr_time - arrival_times.pop(job_id)
    comp_jobs += 1

    # If there are more jobs waiting in queue for stage 2, start the next job
    if len(q2) > 0:
        next_job_id = q2.pop(0)
        process_stage_2(next_job_id)


def main():
    global curr_time

    # This is not an actual event
    event = Event(0, EventType.ARRIVAL, job_id_t(-1))
    es.insert(event)

    start_time = 0
    queue_1_len = 0
    queue_2_len = 0

    q1_freq: defaultdict[int, float] = defaultdict(lambda: 0)
    q2_freq: defaultdict[int, float] = defaultdict(lambda: 0)
    overall_freq: defaultdict[int, float] = defaultdict(lambda: 0)
    while not es.is_empty() and curr_time < SIM_TIME:
        event = es.pop()
        curr_time, event_type, job_id = event.time, event.type, event.job_id
        elapsed = curr_time - start_time
        q1_freq[queue_1_len] += elapsed
        q2_freq[queue_2_len] += elapsed
        overall_freq[queue_1_len + queue_2_len] += elapsed

        queue_1_len = len(q1)
        queue_2_len = len(q2)
        start_time = curr_time

        if event_type == EventType.ARRIVAL:
            arrival()
        elif event_type == EventType.COMPLETE_STAGE_1:
            complete_stage_1(job_id)
        elif event_type == EventType.COMPLETE_STAGE_2:
            complete_stage_2(job_id)
        else:
            assert False, "UNREACHABLE"

    q1_probs = freq_to_prob(q1_freq)
    for length, freq in q1_freq.items():
        print(length, freq)

    print()

    q2_probs = freq_to_prob(q2_freq)
    for length, freq in q2_freq.items():
        print(length, freq)

    print()

    overall_probs = freq_to_prob(overall_freq)
    for length, freq in overall_freq.items():
        print(length, freq)

    prove_johnsons_theorem(q1_probs, q2_probs, overall_probs)

    average_time_in_system = total_sojourn_time / comp_jobs
    print(f"Total jobs inbound: {total_jobs}")
    print(f"Total jobs completed: {comp_jobs}")
    print(f"Average time in system: {average_time_in_system} units of time")


def freq_to_prob(freqs: dict[int, float]) -> dict[int, float]:
    result: dict[int, float] = {}
    total_freq = 0
    for _, freq in freqs.items():
        total_freq += freq

    for lenght, freq in freqs.items():
        result[lenght] = freqs[lenght] / total_freq

    return result


def prove_johnsons_theorem(
    q1_probs: dict[int, float],
    q2_probs: dict[int, float],
    overall_probs: dict[int, float],
):
    result = {}
    for length in overall_probs.keys():
        result[length] = 0
        for i in range(0, length + 1):
            q1_len = i
            q2_len = length - i

            if q1_len in q1_probs and q2_len in q2_probs:
                result[length] += q1_probs[q1_len] * q2_probs[q2_len]

    print()
    for k, v in overall_probs.items():
        print(f"{k}, {v:.5f}, {result[k]:.5f}")


if __name__ == "__main__":
    main()
