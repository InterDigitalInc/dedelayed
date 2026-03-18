
https://github.com/user-attachments/assets/f4e4ab62-aea8-47bf-a2e1-f8822a490541


| Metric              | Remote-only | Remote-predictive | Local-only | Dedelayed (Local + Remote)  |
| ------------------- | :---------: | :---------------: | :--------: | :-------------------------: |
| Always on-time?     |      No     |         No        |     Yes    |             Yes             |
| Sees pedestrians?   |     Yes     |        Yes        |     No     |             Yes             |
| mIoU @ RTT=167 ms   |    0.514    |       0.596       |    0.588   |            0.641            |
| mIoU @ RTT=0 ms     |    0.649    |       0.655       |    0.588   |            0.661            |


*Dedelayed* is always on-time (≤33ms) and sees pedestrians, even under network round-trip time (RTT) of 167 ms.


