# Define processing times for each job operation on each machine
processing_times = {
    "Job 1": {"Machine 1": 2, "Machine 2": 1},  # Job 1: Operation 1 takes 2 hours on Machine 1, 1 hour on Machine 2
    "Job 2": {"Machine 1": 1.5, "Machine 2": 1.5},
    "Job 3": {"Machine 1": 3, "Machine 2": 2},
}

# Define the order of machines
machine_order = ["Machine 1", "Machine 2"]

# Define the starting time of the workday
start_time = 8  # 8 AM

# Function to calculate the completion time of an operation of a job on a machine
def get_completion_time(job, machine, current_time):
    """
    Calculates the completion time of the next operation of a job on a specific machine,
    considering the current time and processing time.

    Args:
        job: The job being processed.
        machine: The name of the machine.
        current_time: The current time in hours.

    Returns:
        The completion time of the next operation on the machine, considering the current time
        and processing time.
    """
    next_operation = len([op for op in processing_times[job].keys() if op <= machine])
    processing_time = processing_times[job][machine]
    return current_time + processing_time, next_operation

# Function to schedule jobs using First Come First Served (FCFS) algorithm
def fcf_schedule(jobs):
    """
    Schedules a list of jobs using the First Come First Served (FCFS) algorithm,
    considering processing times, machine order, and a limited workday duration.

    Args:
        jobs: A list of jobs to be scheduled, where each job is represented as a list of operations.

    Returns:
        A dictionary containing completion times for each operation and the Makespan (total completion time).
    """
    completion_times = {}  # Dictionary to store completion times for each operation
    current_time = start_time

    while any(jobs):
        for job in jobs[:]:
            for machine in machine_order:
                completion_time, next_operation = get_completion_time(job, machine, current_time)
                completion_times[(job, machine)] = completion_time
                current_time = completion_time

            jobs.remove(job)  # Remove the job from the list once completed

    # Calculate Makespan (completion time of the last operation)
    makespan = max(completion_times.values())
    return completion_times, makespan

# Function to schedule jobs using Round Robin (RR) algorithm
def rr_schedule(jobs, time_quantum):
    """
    Schedules a list of jobs using the Round Robin (RR) algorithm,
    considering processing times, machine order, and a limited workday duration.

    Args:
        jobs: A list of jobs to be scheduled, where each job is represented as a list of operations.
        time_quantum: The time quantum for the Round Robin algorithm.

    Returns:
        A dictionary containing completion times for each operation and the Makespan (total completion time).
    """
    completion_times = {}  # Dictionary to store completion times for each operation
    current_time = start_time

    while any(jobs):
        for job in jobs[:]:
            for machine in machine_order:
                completion_time, next_operation = get_completion_time(job, machine, current_time)
                completion_times[(job, machine)] = completion_time
                current_time = completion_time
                if time_quantum is not None:
                    current_time += time_quantum  # Add time quantum

            jobs.remove(job)  # Remove the job from the list once completed

    # Calculate Makespan (completion time of the last operation)
    makespan = max(completion_times.values())
    return completion_times, makespan

# Function to schedule jobs using Shortest Job First (SJF) algorithm
def sjf_schedule(jobs):
    """
    Schedules a list of jobs using the Shortest Job First (SJF) algorithm,
    considering processing times, machine order, and a limited workday duration.

    Args:
        jobs: A list of jobs to be scheduled, where each job is represented as a list of operations.

    Returns:
        A dictionary containing completion times for each operation and the Makespan (total completion time).
    """
    completion_times = {}  # Dictionary to store completion times for each operation
    current_time = start_time

    while any(jobs):
        shortest_job = min(jobs, key=lambda job: sum(processing_times[job].values()))
        for machine in machine_order:
            completion_time, next_operation = get_completion_time(shortest_job, machine, current_time)
            completion_times[(shortest_job, machine)] = completion_time
            current_time = completion_time

        jobs.remove(shortest_job)  # Remove the job from the list once completed

    # Calculate Makespan (completion time of the last operation)
    makespan = max(completion_times.values())
    return completion_times, makespan

# Sample list of jobs to be processed
jobs = ["Job 1", "Job 2", "Job 3"]

rr_jobs = jobs.copy()
sjf_jobs = jobs.copy()

# Schedule jobs using FCFS and print results
fcf_completion_times, fcf_makespan = fcf_schedule(jobs)
print("Completion Times (FCFS):")
sorted_completion_times = sorted(fcf_completion_times.items(), key=lambda x: x[1])
for (job, machine), completion_time in sorted_completion_times:
    print(f"{job} on {machine}: {completion_time}")
print(f"\nMakespan (FCFS): {fcf_makespan}\n")

# Schedule jobs using Round Robin (RR) with a time quantum of 1 hour and print results
rr_completion_times, rr_makespan = rr_schedule(rr_jobs, time_quantum=1)
print("Completion Times (Round Robin):")
sorted_completion_times = sorted(rr_completion_times.items(), key=lambda x: x[1])
for (job, machine), completion_time in sorted_completion_times:
    print(f"{job} on {machine}: {completion_time}")
print(f"\nMakespan (Round Robin): {rr_makespan}\n")

# Schedule jobs using Shortest Job First (SJF) and print results
sjf_completion_times, sjf_makespan = sjf_schedule(sjf_jobs)
print("Completion Times (Shortest Job First):")
sorted_completion_times = sorted(sjf_completion_times.items(), key=lambda x: x[1])
for (job, machine), completion_time in sorted_completion_times:
    print(f"{job} on {machine}: {completion_time}")
print(f"\nMakespan (Shortest Job First): {sjf_makespan}\n")