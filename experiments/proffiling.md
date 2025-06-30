# 🧪 System Profiling for DoE (Design of Experiments)

This document outlines the system characteristics to estimate the maximum number of design evaluations (`N_max`) we can run per overnight batch.

---

## ⚙️ System Configuration

| Parameter         | Value                                  |
|------------------|----------------------------------------|
| **CPU Cores (C)**| 26 logical (13 physical × 2 threads)   |
| **Eval Time (Tₑᵥₐₗ)** | 6.05 seconds per design                |
| **Run Window (H)**| 8 hours (overnight batch)             |
| **Overhead (O)**  | 20% (file I/O, parsing, job scheduling) |

---

## 📈 Formula

\[
N_{\max} = \left\lfloor \frac{C \times (H \times 3600)}{T_{\mathrm{eval}} \times (1 + O)} \right\rfloor
\]

---

## 🔢 Example Calculation

\[
N_{\max} = \left\lfloor \frac{26 \times (8 \times 3600)}{6.05 \times 1.2} \right\rfloor
= \left\lfloor \frac{936,000}{7.26} \right\rfloor \approx 103,076
\]

- **Estimated Capacity:** ~103k design evaluations per night.

---

## 📝 Notes

- Evaluation measured using:  
  ```bash
  time ./run_one_design.sh
