
https://github.com/user-attachments/assets/f4e4ab62-aea8-47bf-a2e1-f8822a490541

Under 167 ms of round-trip delay:

| Remote-only       | Remote-predictive | Local-only          | Dedelayed (Local + Remote) |
|:-----------------:|:-----------------:|:-------------------:|:-------------------:|
| delayed           | delayed           | always on-time      | always on-time      |
| sees pedestrians  | sees pedestrians  | misses pedestrians  | sees pedestrians    |
| 0.514 mIoU@167ms  | 0.596 mIoU@167ms  | 0.588 mIoU@167ms    | 0.641 mIoU@167ms    |
| 0.649 mIoU@0ms    | 0.655 mIoU@0ms    | 0.588 mIoU@0ms      | 0.661 mIoU@0ms      |


