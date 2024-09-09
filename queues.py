# Copyright (C) 2024 Patel, Nimai <nimai.m.patel@gmail.com>
# Author: Patel, Nimai <nimai.m.patel@gmail.com>

from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import math
import random
from enum import Enum, auto
from typing import List, NewType, Optional


job_id_t = NewType("job_id_t", int)


class EventType(Enum):
    ARRIVAL = auto()
    PROCESS_STAGE_1 = auto()
    COMPLETE_STAGE_1 = auto()
    PROCESS_STAGE_2 = auto()
    COMPLETE_STAGE_2 = auto()


@dataclass
class Event:
    time: float
    type: EventType
    job_id: job_id_t


@dataclass
class EventStackNode:
    event: Event
    next: Optional[EventStackNode] = None
    prev: Optional[EventStackNode] = None


@dataclass
class EventStack:
    head: Optional[EventStackNode] = None
    tail: Optional[EventStackNode] = None


# Constants
ARRIVAL_RATE = 1
SERVICE_TIME_STAGE_1 = 3
SERVICE_TIME_STAGE_2 = 4
SIM_TIME = 10000

# State
curr_time: float = 0
q1: List[job_id_t] = []
q2: List[job_id_t] = []
server_1_busy = False
server_2_busy = False
es = EventStack()


# Stats
total_jobs = 0
comp_jobs = 0
total_sojourn_time: float = 0
arrival_times: dict[job_id_t, float] = {}

# Misc.
event_log_file_handle = None


def log_event(job_id: job_id_t, event: EventType, time: float):
    global event_log_file_handle

    if event_log_file_handle == None:
        event_log_file_handle = open("event-log.csv", "w")
        event_log_file_handle.truncate(0)
        event_log_file_handle.write("Job ID, Event Type, Event Time\n")

    event_log_file_handle.write(f"{job_id}, {event.name}, {time}\n")


def exponential_random(mean: float) -> float:
    U = random.random()
    return -math.log(1 - U) / mean


def ES_Is_Empty(es: EventStack):
    return es.head == None


def ES_Insert(es: EventStack, event: Event):
    node = EventStackNode(event)

    if ES_Is_Empty(es):
        es.head = node
        es.tail = node
        return

    assert (
        es.head != None
    ), "head cannot be None, we checked that the queue is not empty"
    if node.event.time <= es.head.event.time:
        node.next = es.head
        es.head.prev = node
        es.head = node
        return

    assert (
        es.tail != None
    ), "tail cannot be None, we checked that the queue is not empty"
    if node.event.time >= es.tail.event.time:
        node.prev = es.tail
        es.tail.next = node
        es.tail = node
        return

    curr = es.head
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


def ES_Pop(es: EventStack):
    assert es.head != None
    event = es.head.event
    es.head = es.head.next

    if es.head:
        es.head.prev = None
    else:
        es.tail = None

    return event


def arrival():
    """Handles job arrival."""
    global total_jobs

    total_jobs += 1
    job_id = job_id_t(total_jobs)
    log_event(job_id, EventType.ARRIVAL, curr_time)
    arrival_times[job_id] = curr_time

    # Schedule the next arrival (Poisson process)
    inter_arrival_time = exponential_random(ARRIVAL_RATE)
    next_arrival_event = Event(
        curr_time + inter_arrival_time, EventType.ARRIVAL, job_id
    )
    ES_Insert(es, next_arrival_event)

    # If server for stage 1 is free, start processing the job
    if server_1_busy:
        q1.append(job_id)
    else:
        process_stage_1(job_id)


def process_stage_1(job_id: job_id_t):
    """Processes a job in stage 1."""
    global server_1_busy
    server_1_busy = True

    log_event(job_id, EventType.PROCESS_STAGE_1, curr_time)

    # Schedule the completion of stage 1
    serv_time = exponential_random(SERVICE_TIME_STAGE_1)
    event = Event(curr_time + serv_time, EventType.COMPLETE_STAGE_1, job_id)
    ES_Insert(es, event)


def complete_stage_1(job_id: job_id_t):
    """Handles the completion of stage 1 for a job."""
    global server_1_busy

    log_event(job_id, EventType.COMPLETE_STAGE_1, curr_time)
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

    log_event(job_id, EventType.PROCESS_STAGE_2, curr_time)

    # Schedule the completion of stage 2
    serv_time = exponential_random(SERVICE_TIME_STAGE_2)
    event = Event(curr_time + serv_time, EventType.COMPLETE_STAGE_2, job_id)
    ES_Insert(es, event)


def complete_stage_2(job_id: job_id_t):
    """Handles the completion of stage 2 for a job."""
    global server_2_busy, total_sojourn_time, comp_jobs

    log_event(job_id, EventType.COMPLETE_STAGE_2, curr_time)
    server_2_busy = False

    # Track total time in the system for the job
    total_sojourn_time += curr_time - arrival_times.pop(job_id)
    comp_jobs += 1

    # If there are more jobs waiting in queue for stage 2, start the next job
    if len(q2) > 0:
        next_job_id = q2.pop(0)
        process_stage_2(next_job_id)


def free_resources():
    if event_log_file_handle != None:
        event_log_file_handle.close()


def sim_run():
    global curr_time

    prev_time = 0
    prev_q1_len = 0
    prev_q2_len = 0

    q1_freq: defaultdict[int, float] = defaultdict(lambda: 0)
    q2_freq: defaultdict[int, float] = defaultdict(lambda: 0)
    overall_freq: defaultdict[int, float] = defaultdict(lambda: 0)

    arrival()
    while not ES_Is_Empty(es) and curr_time < SIM_TIME:
        event = ES_Pop(es)
        curr_time = event.time

        elapsed = curr_time - prev_time
        q1_freq[prev_q1_len] += elapsed
        q2_freq[prev_q2_len] += elapsed
        overall_freq[prev_q1_len + prev_q2_len] += elapsed

        prev_q1_len = len(q1)
        prev_q2_len = len(q2)
        prev_time = curr_time

        if event.type == EventType.ARRIVAL:
            arrival()
        elif event.type == EventType.COMPLETE_STAGE_1:
            complete_stage_1(event.job_id)
        elif event.type == EventType.COMPLETE_STAGE_2:
            complete_stage_2(event.job_id)
        else:
            assert False, "UNREACHABLE"

    return q1_freq, q2_freq, overall_freq


def main():
    random.seed(10)

    q1_freq, q2_freq, overall_freq = sim_run()

    q1_probs = freq_to_prob(q1_freq)
    q2_probs = freq_to_prob(q2_freq)
    overall_probs = freq_to_prob(overall_freq)

    print("Queue 1 Length, Probability")
    for length, freq in q1_probs.items():
        print(length, freq)

    print()

    print("Queue 2 Length, Probability")
    for length, freq in q2_probs.items():
        print(length, freq)

    print()

    print("System Size, Probability")
    for length, freq in overall_probs.items():
        print(length, freq)

    print()

    verify_johnsons_theorem(q1_probs, q2_probs, overall_probs)

    print()

    avg_sojourn_time = total_sojourn_time / comp_jobs
    print(f"Total jobs inbound: {total_jobs}")
    print(f"Total jobs completed: {comp_jobs}")
    print(f"Average time in system: {avg_sojourn_time} units of time")

    free_resources()


def freq_to_prob(freq_dist: dict[int, float]) -> dict[int, float]:
    """Takes dictionary that stores frequency distribution and returns
    dictionary that stores probability distribution"""
    prob_dist: dict[int, float] = {}
    total_freq = 0
    for freq in freq_dist.values():
        total_freq += freq

    for var, freq in freq_dist.items():
        prob_dist[var] = freq_dist[var] / total_freq

    return prob_dist


def verify_johnsons_theorem(
    q1_probs: dict[int, float],
    q2_probs: dict[int, float],
    measu_overall_probs: dict[int, float],
):
    calcu_overall_probs = {}
    for overall_len in measu_overall_probs.keys():
        calcu_overall_probs[overall_len] = 0
        for i in range(0, overall_len + 1):
            q1_len = i
            q2_len = overall_len - i

            if q1_len in q1_probs and q2_len in q2_probs:
                calcu_overall_probs[overall_len] += q1_probs[q1_len] * q2_probs[q2_len]

    print("System Size, Measured Probability, Calculated Probability")
    for k, v in measu_overall_probs.items():
        print(f"{k}, {v:.5f}, {calcu_overall_probs[k]:.5f}")


if __name__ == "__main__":
    main()
