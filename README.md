# Course Project Description
### ECE 493 / ECE 720 — Machine Learning Hardware Systems

---

## Overview

The course project is a substantial team effort centered on a published machine learning accelerator paper. Teams will select a paper, study it in depth, implement the accelerator presented in it, and replicate its key results. The project is designed to give you hands-on experience with the full accelerator design stack, from architecture modeling to RTL implementation and verification to workload bringup on the implemented accelerator.

This is deliberately an ambitious project -- the bar is set high! You are expected to leverage all available productivity tools to implement something meaningful beyond conventional small-scale course projects and have interesting experience to talk about in your future interviews. Strong project outcomes that propose novel ideas/implementations beyond existing literature can also be suitable for publication.

---

## Team Formation

Form a team of **5 students**. You are free to organize the work however your team sees fit. As a suggested (non-mandatory) division of responsibilities, teams may consider the following roles:

| Role | Suggested Responsibility |
|---|---|
| Architect | Functional and performance SW simulator for the accelerator |
| HW Engineers | SystemVerilog RTL implementation |
| Verification Engineer | Unit and integration testing |
| Workload Bring-up | End-to-end workload execution in RTL simulation |

These are suggestions only. Your team may divide work differently as long as all project components are completed and all team members participate equally in the project implementation.

---

## Paper Selection

Select a machine learning accelerator paper published in the last 5–6 years from one of the top-tier conferences/journals in this area. Some examples of these venues are:

- International Symposium on Field-Programmable Gate Arrays (FPGA)
- International Symposium on Field-Programmable Custom Computing Machines (FCCM)
- International Conference on Field-Programmable Logic and Applications (FPL)
- International Conference on Field-Programmable Technology (FPT)
- International Symposium on Computer Architecture (ISCA)
- International Symposium on Microarchitecture (MICRO) 
- International Symposium on High-Performance Computer Architecture (HPCA)
- International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)
- International Design Automation Conference (DAC)
- Design, Automation and Test in Europe Conference (DATE)
- International Conference on Computer-Aided Design (ICCAD)
- Asia and South Pacific Design Automation Conference (ASP-DAC)
- International Conference on Machine Learning and Systems (MLSys)
- Transactions on Computers (TC)
- Transactions on Reconfigurable Technology and Systems (TRETS)

The paper you select should satisfy the following conditions:

- Presents an **entire accelerator architecture** for machine learning inference (not a module or a component of an accelerator)
- Showcases performance results for one or more **end-to-end machine learning workload(s)** that you will aim to replicate

Your paper selection must be approved by the course instructor before work begins. Projects must be unique (i.e., one paper cannot be selected by two teams) and will be approved on a first-come first-serve basis.

Submit the "Team Registration and Paper Selection" online form ([link](https://forms.office.com/pages/responsepage.aspx?id=h1o6cprzIkqSRz_CQMATlmltzCfSgtZOqYV6WTUzW_RUMzgzS0IyWVZFUlRQWVBGVVJFNTRHOFBaRy4u&route=shorturl)) by the registration deadline below. A list of all teams and their approved projects will be maintained [here](https://uofwaterloo-my.sharepoint.com/:x:/g/personal/a2boutro_uwaterloo_ca/IQBY2Jx3KIR8Q4FPdYeVVHwmAXiWCiFK7_QVdNIc3Lw6d5E?e=55dchV). Before submitting a form, make sure that the paper you selected is not already on the approved list.

---

## Project Goals

1. Implement the entire accelerator in **SystemVerilog RTL**.
2. Develop a C++/SystemC/Python functional and performance simulator of the selected accelerator. This can serve as a comparison point and ground truth for RTL simulation. **This is mandatory for ECE 720 students but optional for ECE 493 students. Teams that mix ECE 493 and 720 students will have to deliver this component.** 
3. Build RTL simulation testing infrastructure to verify the functionality of each component of the accelerator (**unit tests**) as well as groups of components and the entire architecture (**integration tests**). You can develop vanilla SystemVerilog testbenches or use frameworks such as cocotb or UVM. You can use the open-source [Verilator](https://github.com/verilator/verilator) simulator or any other RTL simulator you have access to (e.g., Vivado simulator, Questa, VCS).
4. Obtain **area/resource and operating frequency estimates** by implementing the design targeting an FPGA device (through Vivado or Quartus or open-source [VTR](https://github.com/verilog-to-routing/vtr-verilog-to-routing) toolsuites) or as an ASIC (through the open-source [OpenROAD](https://github.com/The-OpenROAD-Project/OpenROAD) flow or other commercial tools you have access to). Deployment of the accelerator on actual FPGA hardware is not required.
5. Bring up an entire workload to be simulated on the implemented accelerator to obtain **performance results**.
6. Compare performance, area, and/or accuracy results to those reported in the paper. Your results are not expected to match the paper 100% (since you will have to make assumptions and fill in gaps in the paper description of the accelerator), but you should be able to reason about possible cause of discrepancies.

---

## Project Deliverables

The project has 5 graded deliverables to be submitted during the term by the deadlines listed below.

### Early-bird Project Selection and Team Registration (2%)

### Project Plan & Weekly Milestones (3%)

A detailed plan of the project is submitted through the project GitLab repo as a `reports/plan.md` Markdown file. This plan should contain:

- Brief overview of project-specific components and different tasks (What will be done?)
- Split of responsibilities and tasks between team members (Who will do it?)
- List of weekly milestones and deliverables that the team is aiming to get done by Sunday of each week of the term (When will it be done?)

### Two Project Progress Reports (10%)

Detailed reports submitted through the project GitLab repo as `reports/progress_1.md` and `reports/progress_2.md` Markdown files. Each report should include:

- Report on the progress of tasks **per team member**
- List of the weekly milestones from the initial project plan (from project start to date) indicating which targets were met, delayed, or skipped with justifications
- Any refinements to the initial plan for the next phase of the project

### Poster Presentation (15%)

The tentative plan is that we will hold a long poster session during the last week of classes in which each team will have a poster describing their project, implementation details, testing strategy, and preliminary results.
This is a great chance for all teams to see what other teams have been cooking throughout the term and exchange knowledge by discussing projects. The course instructor will also be evaluating projects during this poster session through discussions with each team. More details about the logistics of this poster session will be announced later during the term.

### Final Report & Polished Code Base (10%)

An up to 8-page two-column IEEE format ([Overleaf template](https://www.overleaf.com/latex/templates/ieee-conference-template/grfzhhncsfqn)) final project report that has a similar skeleton to a research paper:

- Introduction: why is this project interesting?
- RTL Implementation Details: how the different accelerator components are implemented?
- Testing Methodology: how was testing and verification performed?
- Workloads & Software Simulator: what are the workloads showcased on the accelerator? and, if applicable, a description of the software simulator developed
- Results:
    
    - Comparison between software simulator and RTL simulation results (if applicable)
    - Comparison between obtained and published results
    - Discussion and commentary on the results
- Conclusion & Future Work
- A link to the GitLab repo

The final report should be pushed to the repo as a `reports/final_report.pdf` file.

Your project collateral should be pushed to the repo under the `rtl/`, `testing\`, and `sw\` directories that should include your RTL implementations, any testbenches or testing infrastructure/scripts, and any software collateral you developed for the project (including the software simulator for ECE 720 teams), respectively.

The project repo should be clean with a detailed `README.md` describing the contents of the repo and how to reproduce the reported results (development environment and automation scripts).

> [!IMPORTANT] All team members must contribute to the project by committing and pushing their own code to the repo. You also must have proper commit messages that clearly describe the work done for a specific commit pushed to the repo. THIS WILL BE PART OF YOUR PROJECT EVALUATION.

---

## Deadlines

- **Team registration & project selection:** Tuesday, May 27 @ 11:59 pm (2% early-bird incentive if finalized by Sunday, May 24 @ 11:59 pm)
- **Project plan & weekly milestones:** Sunday, May 31 @ 11:59 pm
- **Project progress report 1:** Friday, June 19 @ 11:59 pm 
- **Project progress report 2:** Friday, July 17 @ 11:59 pm
- **Poster presentation:** TBD (last week of classes)
- **Final paper:** Wednesday, August 19 @ 11:59 pm (Last day of exams)

---

## FAQs

#### Can we form groups of less or more than 5?

The project workload is suitable for 5 students working on it for an entire term. If you decide to have a smaller team, you would be signing up for more work, but this is acceptable. If a team has more than 5 members, additional deliverables or a higher project complexity level will be expected. This will require approval from the course instructor.

#### I am a research graduate student and my research project is relevant to this course project, can I work on a research project that does not fit this project description for the course?

It is possible as long as there is a clear plan on what the project deliverables will be. This will be approved on a case-by-case basis depending on the scope and complexity of the proposed project.

#### Can we implement our project in HLS?

No. The accelerator implementation must be in SystemVerilog RTL.

#### What is a software functional and performance simulator?

A functional and performance simulator is a software code that you write to emulate the architecture of the accelerator (typically done before RTL implementation to rapidly evaluate ideas and explore architecture choices). This can be written in C++, SystemC, Python, or any other programming language. A **functional** simulation means that the operations done by the accelerator are actually computed in software and generate correct results. A **performance** simulator means that it estimates how many cycles a sequence of operations would take when running on the accelerator. This is typically a lot faster than RTL simulation, easier to tweak for exploring different ideas, but less accurate because it is not necessarily written to model cycle-level behavior.

[SystemC](https://systemc.org) is a C++ library that makes modeling hardware in software a lot easier (introducing the notion of clock cycle and semantics of concurrent execution). [This](https://www.youtube.com/watch?v=NCFxBGLB5xs&list=PLcvQHr8v8MQLj9tCYyOw44X1PLisEsX-J) video tutorial series is a beginner-level introduction to SystemC, in case you are curious to learn more about it and use it for this component of the course project. However, feel free not to use it, if you prefer not to. 

#### If we choose an accelerator that interacts with external interfaces (e.g., DDR, HBM, Ethernet, PCIe), how can we implement this?

You can model external interfaces within the testbench. For example, the testbench can act as a memory model that receives read/write requests from the accelerator, handles bookkeeping, and returns responses. You can make reasonable assumptions about the latency (how many cycles before a response is returned) and bandwidth (bus width at a given frequency) of these interfaces. You can also inject randomness to simulate non-deterministic access latencies, as found in real DDR/HBM memory systems.

#### How do we access the tools we need (e.g., Vivado, Quartus) for the project?

You are expected to set up any tools you need on your laptop/PC. Undergraduate labs at UWaterloo-ECE have some of these tools installed and you can use them at your convenience when there are no classes running in the rooms. There are also open-source alternatives linked above that could be easier to set up and use on your own laptop/PC without the need for any licenses. 

#### Can we select a paper for which the RTL code is open-source?

Only if you have a very clear plan on how you want to extend the open-source with something new as the project deliverable. This will be approved on a case-by-case basis.