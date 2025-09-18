import matplotlib.pyplot as plt

threads = [1, 2, 4, 8, 16, 32, 64, 127, 128]

# Static scheduling (s1 and s4)
best_s1 = [0.77, 1.61, 2.43, 5.33, 8.74, 13.72, 21.14, 23.61, 23.90]
worst_s1 = [0.51, 1.13, 1.83, 3.31, 5.22, 7.81, 11.32, 19.31, 23.11]

best_s4 = [0.78, 1.88, 2.44, 5.37, 9.54, 15.92, 23.91, 24.41, 24.15]
worst_s4 = [0.78, 1.25, 1.81, 3.32, 5.17, 8.18, 12.17, 21.30, 23.67]

# Dynamic scheduling (d1 and d4)
best_d1 = [0.77, 1.48, 2.05, 3.16, 4.87, 5.21, 7.08, 7.48, 7.55]
worst_d1 = [0.77, 1.36, 1.78, 2.68, 3.84, 4.44, 5.37, 7.04, 7.33]

best_d4 = [0.79, 1.82, 2.78, 5.16, 9.04, 11.65, 14.77, 12.78, 12.89]
worst_d4 = [0.79, 1.54, 2.33, 3.89, 6.06, 7.86, 9.08, 11.76, 11.37]

plt.figure(figsize=(12, 7))

# Static best
plt.plot(threads, best_s1, 'b-o', label='Static s1 Best')
plt.plot(threads, best_s4, 'b--o', label='Static s4 Best')

# Static worst
plt.plot(threads, worst_s1, 'b-x', label='Static s1 Worst')
plt.plot(threads, worst_s4, 'b--x', label='Static s4 Worst')

# Dynamic best
plt.plot(threads, best_d1, 'r-o', label='Dynamic d1 Best')
plt.plot(threads, best_d4, 'r--o', label='Dynamic d4 Best')

# Dynamic worst
plt.plot(threads, worst_d1, 'r-x', label='Dynamic d1 Worst')
plt.plot(threads, worst_d4, 'r--x', label='Dynamic d4 Worst')

plt.xlabel('Number of Threads')
plt.ylabel('Performance (GFLOPS)')
plt.title('Static vs Dynamic Scheduling Performance (Best & Worst)')
# plt.xscale('log', base=2)
plt.xticks(threads, labels=threads)
plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.legend(loc='upper left', fontsize='small', ncol=2)
plt.tight_layout()
plt.show()
plt.savefig('scheduling_performance_2d.png')
