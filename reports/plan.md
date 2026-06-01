# ECE493/720 Course Project Plan - Team 19
Team Members: Fred Hunag (j462huan@uwaterloo.ca), Emon Sarkar (esarkar@uwaterloo.ca), Mahdi Hassen (mhassen@uwaterloo.ca), Sara Ahmad (s92ahmad@uwaterloo.ca), Adil Kazimov (akazimov@uwaterloo.ca)

### Project Overview
Recreate the GoSPA sparse CNN accelerator from the paper [**GoSPA: An Energy-efficient High-performance Globally Optimized SParse Convolutional Neural Network Accelerator**](https://ieeexplore.ieee.org/document/9499915), with evaluation of the expected performance demonstrated in the paper. The general breakdown of tasks are as follows:
1. Implementation of the GoSPA architecture for 2D convolution acceleration as proposed in the paper.
    * Synthesizable RTL implementation with minor modification to the memory hierarchy (fixed latency)
2. Software simulator for functional and performance validation
3. Develop verification infrastructure and bring up real-world CNN workloads

### The GoSPA Architecture consistent of 3 key pieces:

1. APU Stage 1: ID Genereration

    * Taking in a Matrix in CSR format, this stage assigns each non-zero element custom positions (CID and PID)  

2. APU Stage 2: PE Assignment
    * Stage 2 assigns each element, based on it's PID/CID into a PE

3. PE: Processing element
    * When a receiving an element, based on the PID and CID, the computation is reordered in a manner that allows for high data resuse


### Task Breakdown By Member & Weekly Milestones
To achieve the weekly milestones, we break down the tasks for each team member. Note that at the start of the project, Fred, Mahdi and Sara will be focusing on the SW model while Emon and Adil will get into the HW implementation. Based on the milestones, we expect to **finish the full SW model before July 5**, at which the SW team members will turn to facilitate the finalization of RTL implementation and start system integration & testing. The following table summarizes the specific breakdown for each team member, based on the dates where we expect each milestone to be accomplished. 

### Team Milestones

June 21: Finish Functional Model, APU Stage 1

July 5: Finish SW Perf Model

July 12: Finish and Verify APU Stage 2 (full APU)

July 19: Finish PE

July 26: Finish GoSPA Top Level

August 2: Finish Full CNN with GoSPA at each layer

August 9: Complete verification, gather metrics

August 16: Complete report and presentation

### Person Specific

| Week| Fred | Mahdi | Sara  | Emon  | Adil  |
|---|---|---|---|---|---|
| June 7  |  Research perf model requiremnts, determine abstarction level |  Finish block diagram for HW design | Breakdown the implementation into submodules to determine the functional tests for perf model | Understand HW constraimts, coordinate with Adil about module communication protocols (type of AXI, which modules), setup enviorment on lab PC create fifo.sv | Understand HW constraints, design memory.sv files (sram, dram)|
| June 14 |  makes skeleton of perf model defining level of abstraction| Working on Functional Model |model hardware components on specified abstraction layer | create csr_decode.sv, zero_act.sv,  | create position_encode.sv, idgen.sv |
|  June 21 |  Working on Perf Model | Finish Functional Model  | Working on Perf Model |  merge into APU_stage1.sv with Adil's work and the FIFO's | create router.sv |
|June 28 (midterm week)|--|--|--|--|--|
|  July 5 | Finish Perf Model  | Make intermediate layers for full CNN testing (maxpooling, fully connected) in HW  | Finish Perf Model   |  create apu.sv with router & FIFO's | begin PE, start pe_rf.sv (pe register files), pe_accum_file.sv (accume reg file)|
|  July 12 | start PE full TB |  Continue working intermediate hardware modules | Work on other intermediate HW modules|  APU full TB | pe_selector.sv, start PE full TB  |
|  July 19 | Work with Adil on PE TB  |  Finish all intermediate HW | Compare perf APU with HW APU  |  Help finish PE remaining modules|  finsh PE full TB |
|  July 26 | Work on loading model onto simulation  | Work on loading model onto simulation  |  Compare PE perf with PE SW | Finish Top level module | Begin Top level module |
|  Aug 2 | Help make TB, run figure out porting CNNs to HW  |  Help make TB, run figure out porting CNNs to HW | Make baseline model, no GoSPA, for comparison Naive CNN approach | Make TB for Top level | Finish Top Level, make TB|
|  Aug 9 | Verification last min testing  |  Verification last min testing | Verification last min testing  |  Run synth resource estimation & Power with OpenLane/LibreLane| Check timing on Vivado|
|  Aug 16 (Everyone work on report for their parts) |   |   |   |   |   |

