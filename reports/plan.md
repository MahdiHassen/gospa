# ECE493/720 Course Project Plan - Team 19
Team Members: Fred Hunag (j462huan@uwaterloo.ca), Emon Sarkar (esarkar@uwaterloo.ca), Mahdi Hassen (mhassen@uwaterloo.ca), Sara Ahmad (s92ahmad@uwaterloo.ca), Adil Kazimov (akazimov@uwaterloo.ca)

### Project Overview
Recreate the GoSPA sparse CNN accelerator from the paper [**GoSPA: An Energy-efficient High-performance Globally Optimized SParse Convolutional Neural Network Accelerator**](https://ieeexplore.ieee.org/document/9499915), with evaluation of the expected performance demonstrated in the paper. The general breakdown of tasks are as follows:
1. Implementation of the GoSPA architecture for 2D convolution acceleration as proposed in the paper.
    * Synthesizable RTL implementation with minor modification to the memory hierarchy
2. Software simulator for functional and performance validation
3. Develop verification infrastructure and bring up real-world CNN workloads
4. (To be added...)


### Team Weekly Milestones
As a team, we propose the following list of weekly milestones that aim to accomplish by the Sunday of each week.

**June 7** : bring up; understand the paper more e.t.c
* A Research how we want to develop the performance simulator
* Determine all the high-level and medium level modules required for the accelerator
    - create block design

**June 14** : CSR helper function
* Create the performance simulator block design
* Define latency-insensitive communication protocol
* **Project progress report 1: Friday, June 19 @ 11:59 pm**

**June 21** : implement cid/pid generator
* Implement the performance simulator block design
* Finish combinational CSR->PID/CID blocks

**To be added...............................**

### Task Breakdown By Member
To achieve the weekly milestones listed above, we break down the tasks for each team member. Note that at the start of the project, Fred, Mahdi and Sara will be focusing on the SW model while Emon and Adil will get into the HW implementation. Based on the milestones above, we expect to finish the full SW model before July 5, at which the SW team members will turn to facilitate the finalization of RTL implementation and start system integration & testing. The following table summarizes the specific breakdown for each team member, based on the dates where we expect each milestone to be accomplished. 

|   |  Fred | Mahdi  | Sara  | Emon  | Adil  |
|---|---|---|---|---|---|
| June 7  |  A |  B | C  | D  | E  |
| June 14 |   |   |   |   |   |
|  June 21 |   |   |   |   |   |
|  June 28 |   |   |   |   |   |
|  July 5 |   |   |   |   |   |
|  July 12 |   |   |   |   |   |
|  July 19 |   |   |   |   |   |
|  July 26 |   |   |   |   |   |
|  Aug 2 |   |   |   |   |   |
|  Aug 9 |   |   |   |   |   |
|  Aug 16 |   |   |   |   |   |

