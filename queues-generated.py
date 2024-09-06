import math
import random
import heapq  # To manage events in chronological order

random.seed(10)

# Citation: https://hpaulkeeler.com/simulating-poisson-random-variables-direct-method/
def poisson_random(l):
    N = 0
    S = 0
    while S < 1:
        U = random.random()
        E = -math.log(U) / l
        N += 1
        S += E
    return N


# Parameters
ARRIVAL_RATE = 3  # Average jobs arriving per unit time
SERVICE_TIME_STAGE_1 = 5  # Average service time for first queue
SERVICE_TIME_STAGE_2 = 7  # Average service time for second queue
SIM_TIME = 50  # Total simulation time

# Priority queue to manage events
event_queue = []

# To keep track of jobs waiting to enter the second stage
queue_stage_2 = []

# Current time in the simulation
current_time = 0

# Server states (True = Busy, False = Idle)
server_stage_1_busy = False
server_stage_2_busy = False

# Statistics to track
total_jobs = 0
total_time_in_system = 0


def schedule_event(event_time, event_type, job_id):
    """Schedules an event by adding it to the event queue (min-heap)."""
    heapq.heappush(event_queue, (event_time, event_type, job_id))


def arrival():
    """Handles job arrival."""
    global total_jobs, server_stage_1_busy

    total_jobs += 1
    job_id = total_jobs
    print(f"Job {job_id} arrives at time {current_time:.2f}")

    # Schedule the next arrival (Poisson process)
    inter_arrival_time = poisson_random(ARRIVAL_RATE)
    schedule_event(current_time + inter_arrival_time, "arrival", None)

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
    schedule_event(current_time + service_time_1, "complete_stage_1", job_id)


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
    schedule_event(current_time + service_time_2, "complete_stage_2", job_id)


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


# Schedule the first job arrival to kick off the simulation
schedule_event(0, "arrival", None)

# Run the simulation loop
while event_queue and current_time < SIM_TIME:
    # Get the next event in the event queue (min-heap)
    current_time, event_type, job_id = heapq.heappop(event_queue)

    # Handle the event based on its type
    if event_type == "arrival":
        arrival()
    elif event_type == "complete_stage_1":
        complete_stage_1(job_id)
    elif event_type == "complete_stage_2":
        complete_stage_2(job_id)

# Print final statistics
average_time_in_system = total_time_in_system / total_jobs if total_jobs > 0 else 0
print(f"\nTotal jobs processed: {total_jobs}")
print(f"Average time in system: {average_time_in_system:.2f} units of time")
