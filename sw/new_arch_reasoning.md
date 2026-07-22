## How the utilization is essentially calculated
### 1. Old architecture (--arch channel)
In our old architecture (union WSP), we assumed that PE is always the performance bottleneck (it's not always the case, though). With this assumption, the utilization was calculated as `useful MACs/total PE cycles`. Expanded to multiple PEs each with multipler lanes, for a single pass of compute this is $\frac{\sum_{i}useful\_MACs(PE_i)}{\sum_j pass\_cycles(PE_j)} (Eq.1)$. Furthermore, assuming the network has weight density `dw`, since the total number of (CID, PID) pairs sent to a PE's FIFO-Bs is `n_pairs` and each pair can at the maximum be used in all `M` lanes, the total useful MACs across all PEs is approximately `M * N_PE * dw * n_pairs`. If we assume PE is always the bottleneck, then the pass cycles is determined by the lane that takes the longgest time to execute - assume this time is `pe_cycles`. Then Eq.1 can be rewritten as $\frac{M \cdot N\_PE \cdot dw \cdot n\_pairs}{M \cdot N\_PE \cdot pe\_cycles} = \frac{dw \cdot n\_pairs}{ pe\_cycles}(Eq.2)$. To further generalize Eq.2, note that the execution of each lane is determined by the weight density `dw`, where each weight (i.e., entry in WSP) has a probability of `1 - dw` of being zero. Thus, the probability that a unioned WSP over `M` lanes has a zero entry at a certian PID is $(1-(1-dw)^M)$. Given `n_pairs` input activations, the expected $pe\_cycle= (1-(1-dw)^M) \cdot n\_pairs$. Plug into Eq.2: $util = \frac{dw}{1-(1-dw)^M}$. The implication is that with our `M = 4` and a moderate `dw` (e.g., 0.5), the denominator will be very close to 1, which makes the util almost equal to dw; with smaller `dw` (e.g., 0.1/0.2), the denominator is smaller than 1, which drives util slightly higher than `dw`. This fits the sweep recorded in `sw/union_wsp_util.csv`.

### 2. New architecture, version 1
In the first draft of our new architecture, we changed the datapath but not how the utilization is calculated. The general formula is still `useful MACs/total PE cycles` assuming PE is the bottleneck. In this version, each PE holds a single WSP, so the number of `useful MACs` has changed to $N\_PE \cdot dw \cdot n\_pair$ (makes sense because the number of weights stored in each PE per pass has redeced `M`x), and `total PE cycles` also differs from the old arch. Now on average a (PID, CID) pair gets into a PE simply with the probability `dw`. Once it gets in, the PE completes its MAC in a cycle. So the denominator is now $N\_PE \cdot dw \cdot n\_pair$, which cancels perfectly with the numerator and leaves util approximately 1. It seems perfect, but it's wrong because in most cases PE is not the bottleneck. So it's meaningless to base the util on the pe_cycles given that the bottleneck is something else, where the PEs would just spend time idling. 

### 3. New architecture, version 2
To address this issue, since we already account for the bottleneck stage, we simply change the denominator in the util formula to whatever the bottleneck is. It turns out that stage 2 is most commonly the bottleneck. Basing the util on stage 2 cycles, the util becomes $\frac{N\_PE \cdot dw \cdot n\_pairs}{M \cdot N\_PE \cdot stage2\_cycles} = \frac{dw \cdot n\_pairs}{M \cdot ceil(n\_pairs/M)} (Eq.3)$ where the bottom part is the equation for stage 2 cycles when we take at most M elements out of FIFO-A each cycle. Note that $ceil(n\_pairs/M) \ge n\_pairs/M$ and that meakes the whole util $\le dw$ -- we are back to the original spot. 

You might wonder "what if the bottleneck is not stage 2"? The answer is that the equation for stage 2 does not change no matter what the bottleneck is. So if it's not stage 2, it's something larger, which only makes the util smaller. i.e., the util in this version is systematically bounded by `dw`.

### 4. New architecture, version 3 (current)
An apparent potential solution to the issue is to reduce the stage 2 cycles, which is the lower bound of the denominator in Eq.3. One of the ways to do so is increasing the batch size we take pairs out of FIFO-As. Thus we introduced `--stage2-batch`. Assume that in a pass this batch size is `B`. Now if stage 2 is the bottleneck, we plug into Eq.3 and get $util = \frac{dw \cdot n\_pairs}{M \cdot ceil(n\_pairs/B)} \le dw \cdot \frac{B}{M}$. It scales util up and stage 2 cycles down both by $\frac{B}{M}$ , but it only works when stage 2 is still the bottleneck, i.e., when $ceil(n\_pairs / B) \ge max(stage1, pe, mem)$. This constraint gives the optimal selection of `B`: $B^*=max(1, ceil(\frac{n\_pairs}{max(stage1, pe, mem)}))$. That is, `B*` is the narrowest B that makes Stage 2 stop being the bottleneck. If $pe\_cycles = max(stage1, pe, mem)$, then $B^*=\frac{n\_pairs}{pe\_cycles} \approx \frac{M}{dw}$ because $pe\_cycles \approx \frac{dw \cdot n\_pairs}{M}$. With this mechanism there's been an increase in the util. Below is a run with `--da 0.5 --dw 0.5`:

```
------------------------------------------------------------------
Layer           Cycles   Lat(ms)    Util   Imbal  Bottleneck    B*
------------------------------------------------------------------
conv1        1,013,212     1.013   0.865   1.121      stage1     8
conv2        4,557,610     4.558   0.776   1.285          pe    10
conv3        1,765,361     1.765   0.667   1.476          pe     8
conv4        2,640,810     2.641   0.667   1.475          pe     8
conv5        1,762,325     1.762   0.668   1.474          pe     8
fc6          2,746,611     2.747   0.107   1.338         mem     1
fc7          2,098,201     2.098   0.062   1.680      stage1     1
fc8            512,314     0.512   0.062   1.678         mem     1
------------------------------------------------------------------
TOTAL       17,096,444    17.096   0.523   1.441                10
------------------------------------------------------------------
```

Two potential issues I noticed:

1. With stage 2 out of the way, we can see that for conv layers stage 1 occasionally becomes the bottleneck now.
2. IDK why but besides the util, there has also been an increase in the load imbanlancing. On the layers where PE is the bottleneck, I believe this is the main reason we can't reach ~100%. 