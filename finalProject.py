# Define processing times for each job type on each machine (Saw, Drill)
processing_times = {
    "Job 1": (2, 1),  # Job 1 takes 2 hours on Saw and 1 hour on Drill
    "Job 2": (1.5, 1.5),
    "Job 3": (3, 2),
}

# Define the order of machines (always Saw first, then Drill)
machine_order = ["Saw", "Drill"]

# Define the starting time of the workday
start_time = 8  # 8 AM (in hours)

# Function to calculate the completion time of a job on a machine
def get_completion_time(job_type, machine, current_time):
    """
    Calculates the completion time of a job on a specific machine, considering
    the current time and processing time.

    Args:
        job_type: The type of job being processed (e.g., "Job 1").
        machine: The name of the machine (either "Saw" or "Drill").
        current_time: The current time in hours.

    Returns:
        The completion time of the job on the machine, considering the current time
        and processing time.
    """
    processing_time = processing_times[job_type][machine_order.index(machine)]
    return current_time + processing_time

# Function to schedule jobs using First Come First Served (FCFS) algorithm
def fcf_schedule(jobs):
    """
    Schedules a list of jobs using the First Come First Served (FCFS) algorithm,
    considering processing times, machine order, and a limited workday duration.

    This function iterates through the list of jobs and processes them on machines
    in the defined order (Saw first, then Drill). It calculates the completion time
    for each job on each machine based on the current time and processing time.

    The function also considers a limited workday duration. If a job's completion
    time would exceed the workday end (defined by start_time and a workday duration
    of 8 hours), a warning message is printed, and the completion time is set to the
    end of the workday.

    Args:
        jobs: A list of job types to be scheduled.

    Returns:
        A tuple containing a dictionary of completion times for each job, the Makespan
        (total completion time), and the sequence of jobs processed.
    """
    completion_times = {}  # Dictionary to store completion times for each job
    current_time = start_time
    current_machine = machine_order[0]  # Start with the first machine (Saw)
    sequence = []  # List to store the sequence of jobs processed

    for job in jobs:
        completion_time = get_completion_time(job, current_machine, current_time)

        # Check if workday is exceeded
        if completion_time > start_time + 8:  # Workday ends 8 hours after start_time
            print(f"Warning: Workday exceeded for job {job}")
            completion_time = start_time + 8  # Set completion time to workday end

        completion_times[job] = completion_time
        sequence.append(job)

        # Update current time and machine for the next job
        current_time = completion_time
        current_machine = machine_order[1] if current_machine == machine_order[0] else machine_order[0]

    # Calculate Makespan (completion time of the last job)
    makespan = max(completion_times.values())
    return completion_times, makespan, sequence

# Function to schedule jobs using Round Robin (RR) algorithm
def rr_schedule(jobs, time_quantum):
    """
    Schedules a list of jobs using the Round Robin (RR) algorithm,
    considering processing times, machine order, and a limited workday duration.

    This function iterates through the list of jobs and processes them on machines
    in the defined order (Saw first, then Drill) with a specified time quantum.
    It calculates the completion time for each job on each machine based on the
    current time, processing time, and the time quantum.

    The function also considers a limited workday duration. If a job's completion
    time would exceed the workday end (defined by start_time and a workday duration
    of 8 hours), a warning message is printed, and the completion time is set to the
    end of the workday.

    Args:
        jobs: A list of job types to be scheduled.
        time_quantum: The time quantum for the Round Robin algorithm.

    Returns:
        A tuple containing a dictionary of completion times for each job, the Makespan
        (total completion time), and the sequence of jobs processed.
    """
    completion_times = {}  # Dictionary to store completion times for each job
    current_time = start_time
    current_machine = machine_order[0]  # Start with the first machine (Saw)
    sequence = []  # List to store the sequence of jobs processed

    while jobs:
        for job in jobs[:]:
            completion_time = get_completion_time(job, current_machine, current_time)
            if completion_time > start_time + 8:  # Workday ends 8 hours after start_time
                print(f"Warning: Workday exceeded for job {job}")
                completion_time = start_time + 8  # Set completion time to workday end
            completion_times[job] = completion_time
            sequence.append(job)
            current_time = min(completion_time, start_time + 8)  # Update current time
            current_machine = machine_order[1] if current_machine == machine_order[0] else machine_order[0]  # Switch machine
            jobs.remove(job)  # Remove the job from the list once completed

            if time_quantum is not None and len(jobs) > 0:  # Check for time quantum and remaining jobs
                current_time += time_quantum  # Add time quantum to the current time

    # Calculate Makespan (completion time of the last job)
    makespan = max(completion_times.values())
    return completion_times, makespan, sequence

# Function to schedule jobs using Shortest Job First (SJF) algorithm
def sjf_schedule(jobs):
    """
    Schedules a list of jobs using the Shortest Job First (SJF) algorithm,
    considering processing times, machine order, and a limited workday duration.

    This function iterates through the list of jobs and processes them on machines
    in the defined order (Saw first, then Drill) based on their processing times.
    It calculates the completion time for each job on each machine based on the
    current time and processing time, scheduling the shortest jobs first.

    The function also considers a limited workday duration. If a job's completion
    time would exceed the workday end (defined by start_time and a workday duration
    of 8 hours), a warning message is printed, and the completion time is set to the
    end of the workday.

    Args:
        jobs: A list of job types to be scheduled.

    Returns:
        A tuple containing a dictionary of completion times for each job, the Makespan
        (total completion time), and the sequence of jobs processed.
    """
    completion_times = {}  # Dictionary to store completion times for each job
    current_time = start_time
    current_machine = machine_order[0]  # Start with the first machine (Saw)
    sequence = []  # List to store the sequence of jobs processed

    # Sort jobs based on processing times
    jobs.sort(key=lambda job: processing_times[job][machine_order.index(current_machine)])

    for job in jobs:
        completion_time = get_completion_time(job, current_machine, current_time)

        # Check if workday is exceeded
        if completion_time > start_time + 8:  # Workday ends 8 hours after start_time
            print(f"Warning: Workday exceeded for job {job}")
            completion_time = start_time + 8  # Set completion time to workday end

        completion_times[job] = completion_time
        sequence.append(job)

        # Update current time and machine for the next job
        current_time = completion_time
        current_machine = machine_order[1] if current_machine == machine_order[0] else machine_order[0]

    # Calculate Makespan based on the sorted jobs
    makespan = max(completion_times.values(), default=0)  # Handle empty completion_times
    
    return completion_times, makespan, sequence

# Sample list of jobs to be processed
jobs = ["Job 1", "Job 2", "Job 3"]

# Make a copy of the jobs list before passing it to RR and SJF functions
rr_jobs = jobs.copy()
sjf_jobs = jobs.copy()

# Schedule jobs using FCFS and print results
fcf_completion_times, fcf_makespan, fcf_sequence = fcf_schedule(jobs)
print("Completion Times (FCFS):")
for job in fcf_sequence:
    print(f"{job}: {fcf_completion_times[job]}")
print(f"\nMakespan (FCFS): {fcf_makespan}")
print(f"Sequence (FCFS): {fcf_sequence}\n")

# Schedule jobs using Round Robin (RR) with a time quantum of 1 hour and print results
rr_completion_times, rr_makespan, rr_sequence = rr_schedule(rr_jobs, time_quantum=1)
print("Completion Times (Round Robin):")
for job in rr_sequence:
    print(f"{job}: {rr_completion_times[job]}")
print(f"\nMakespan (Round Robin): {rr_makespan}")
print(f"Sequence (Round Robin): {rr_sequence}\n")

# Schedule jobs using Shortest Job First (SJF) and print results
sjf_completion_times, sjf_makespan, sjf_sequence = sjf_schedule(sjf_jobs)
print("Completion Times (Shortest Job First):")
for job in sjf_sequence:
    print(f"{job}: {sjf_completion_times[job]}")
print(f"\nMakespan (Shortest Job First): {sjf_makespan}")
print(f"Sequence (Shortest Job First): {sjf_sequence}\n")
