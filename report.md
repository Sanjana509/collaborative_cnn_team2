Report: Cross Dataset CNN Collaboration

**Models Used and Dataset Descriptions**

User 1 (Sanjana509)-
    Base Model- MobileNetV2
    Dataset- Cat & Dog Dataset (Tong Python)

User 2 (Tanya1072)-
    Base Model- MobileNetV2 with Dropout
    Dataset- Dogs vs Cats Redux



**Metrics on both datasets**

Summarization table for performance of both Model V1 and improved Model V2-

| Model | Trained On | Tested On | Accuracy |
| :--- | :--- | :--- | :--- |
| V1 | User 1 Data | User 1 Data (Train Metrics) | 97.56% |
| V1 | User 1 Data | User 2 Data (Cross Test) | 97.15% |
| V2 | User 2 Data | User 2 Data (Train Metrics) | 96.23% |
| V2 | User 2 Data | User 1 Data (Cross Test) | 98.23% |



**Observations on generalization and domain shift**

We observed that the two separate datasets were quite similar, which helped the model generalize well across them. With the added dropout layer, Model V2 showed the highest and most reliable accuracy.”
