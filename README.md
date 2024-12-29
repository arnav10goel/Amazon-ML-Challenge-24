# Amazon ML Challenge - Team DBkaScam

This repository contains the code and resources used by **Team DBkaScam** for the [Amazon ML Challenge 2024](https://unstop.com/hackathons/amazon-ml-challenge-amazon-1100713). The team consisted of the following members:
1. [Arnav Goel](https://arnav10goel.github.io/)
2. [Medha Hira](https://medhahira.github.io/)
3. [Mihir Aggarwal](https://www.linkedin.com/in/mihir-agarwal-33b913188/?originalSubdomain=in)
4. [AS Poornash](https://www.linkedin.com/in/a-s-poornash-4973a2240/)

We were ranked among the **Top 10 teams** on the leaderboard and had the honor of presenting our solution to **Amazon Scientists** in the Grand Finale, where we secured an impressive **6th Rank**. Here is a viewing link to our final presenatation: [Link](https://www.canva.com/design/DAGRau30tRI/06v7kPdBwb99GDjsiv1fcg/edit?utm_content=DAGRau30tRI&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

### Overview

In this challenge, our goal was to create an efficent machine learning model to extract entity values from images. Our solution presents an ensemble over open-source vision-language models such as `MiniCPM-2.6` and `Qwen2-VL-7b`, where each component's OCR capabilities are enhanced through a distinct prompting framework.

## Dataset

- We were supposed to utilize the dataset provided by Amazon which included around `~230,000` training images and `~130,000` test images. The dataset is available at the following link: [Link](https://www.kaggle.com/datasets/abhishekgautam12/amazon-ml-challenge-2024).
- Notably, the size of the training dataset mandated using downsampling and EDA methods to get smaller subsets for efficiently performing supervised fine-tuning and curating few-shot exemplars.

## Solution Architecture
<img width="532" alt="Screenshot 2024-11-05 at 2 04 33 AM" src="https://github.com/user-attachments/assets/3bd6ec5c-85f7-41df-a5b6-c7129c320b8f">

Our approach was designed to optimize accuracy and ensure generalization. Thus, we picked a voting-ensemble based approach which comprised of the two models shown above. We detail the strategies for each leg of the ensemble below:

### Zero-Shot Prompting (ZSP)
<img width="479" alt="Screenshot 2024-11-05 at 1 48 05 AM" src="https://github.com/user-attachments/assets/28f55996-aff3-47c5-b13d-599978f9984f">

The image above shows the zero-shot curated prompt we used to prompt `MiniCPM-2.6` with a product image and extract the necessary entity value. The format is made specific to reduce model hallucinations and make post-processing easier.

### Dynamic Few-Shot Learning (FSL)
<img width="479" alt="Screenshot 2024-11-05 at 1 48 51 AM" src="https://github.com/user-attachments/assets/2ff2e278-d3ee-4c2a-8f1c-f87869c0de35">

We employed few-shot learning to improve upon frequently-observed model mistakes in our ZSP approach. Thus we curated a pool of few-shot exemplars consisting of frequent model errors segregated by category. At inference time, we sampled based on the input image and provided three exemplars to better augment the model output. The prompt for the same is shared above.

### Supervised Fine-Tuning (SFT)
- We utilised [LLaMa-Factory](https://github.com/hiyouga/LLaMA-Factory) (Zheng et al., 2024) for performing parameter-efficient SFT on Qwen2-VL-7B using 8-bit QLoRA.
- We fine-tuned `Qwen2-VL-7b` on 150000 samples with a batch size of 16 for 1 epoch for this experiment.

### Post-Processing
Post-processing was an important part of our solution to ensure handling of edge cases and adherance to guidelines.

1. **Handling Edge Cases in Data:**  
   - Fractions and mixed fractions in images are processed using regex expressions to convert them into decimals.  
   - Symbols like single (') and double (") quotes, typically representing feet and inches, are standardized to match the training set mapping.

2. **Managing Ranges and Unknown Symbols:**  
   - For data ranges (e.g., `a-b`), higher values are selected based on predefined guidelines.  
   - Symbols not listed in the reference appendix are removed using instruction-tuned few-shot learning and rule-based algorithms.  

## Results
<img width="453" alt="Screenshot 2024-12-30 at 4 22 46 AM" src="https://github.com/user-attachments/assets/2eaee579-a0c9-4daf-95d5-ed7c70d80d36" />

The results table shows the various model combinations tried by us and the final results achieved by our best performing model.
