# Define processing times for each table type on each machine (Saw, Drill)
processing_times = {
    "Table A": (2, 1),  # Table A takes 2 hours on Saw and 1 hour on Drill
    "Table B": (1.5, 1.5),
    "Table C": (3, 2),
}

# Define the order of machines (always Saw first, then Drill)
machine_order = ["Saw", "Drill"]

# Define the starting time of the workday
start_time = 8  # 8 AM (in hours)

# Function to calculate the completion time of a table on a machine
def get_completion_time(table_type, machine, current_time):
    """
    Calculates the completion time of a table on a specific machine, considering
    the current time and processing time.

    Args:
        table_type: The type of table being processed (e.g., "Table A").
        machine: The name of the machine (either "Saw" or "Drill").
        current_time: The current time in hours.

    Returns:
        The completion time of the table on the machine, considering the current time
        and processing time.
    """
    processing_time = processing_times[table_type][machine_order.index(machine)]
    return current_time + processing_time

# Function to schedule tables using First Come First Served (FCFS) algorithm
def fcf_schedule(tables):
    """
    Schedules a list of tables using the First Come First Served (FCFS) algorithm,
    considering processing times, machine order, and a limited workday duration.

    This function iterates through the list of tables and processes them on machines
    in the defined order (Saw first, then Drill). It calculates the completion time
    for each table on each machine based on the current time and processing time.

    The function also considers a limited workday duration. If a table's completion
    time would exceed the workday end (defined by start_time and a workday duration
    of 8 hours), a warning message is printed, and the completion time is set to the
    end of the workday.

    Args:
        tables: A list of table types to be scheduled.

    Returns:
        A tuple containing a dictionary of completion times for each table and 
        the Makespan (total completion time).
    """
    completion_times = {}  # Dictionary to store completion times for each table
    current_time = start_time
    current_machine = machine_order[0]  # Start with the first machine (Saw)

    for table in tables:
        completion_time = get_completion_time(table, current_machine, current_time)

        # Check if workday is exceeded
        if completion_time > start_time + 8:  # Workday ends 8 hours after start_time
            print(f"Warning: Workday exceeded for table {table}")
            completion_time = start_time + 8  # Set completion time to workday end

        completion_times[table] = completion_time

        # Update current time and machine for the next table
        current_time = completion_time
        current_machine = machine_order[1] if current_machine == machine_order[0] else machine_order[0]

    # Calculate Makespan (completion time of the last table)
    makespan = max(completion_times.values())
    return completion_times, makespan

# Sample list of tables to be processed
tables = ["Table A", "Table B", "Table C"]

# Schedule tables using FCFS and print results
completion_times, makespan = fcf_schedule(tables)
print("Completion Times:")
for table, time in completion_times.items():
    print(f"{table}: {time}")
print(f"\nMakespan (Total Completion Time): {makespan}")
