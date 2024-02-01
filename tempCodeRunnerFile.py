print("\nStatistics:")
for func_name, stats in statistics.items():
    print(f"{func_name}:")
    for stat, value in stats.items():
        print(f"{stat}: {value}")
    print()
