# Hate Speech Intensity Prediction 

This repository contains code for studying **hate speech intensity and escalation**
in Twitter reply chains using temporal, contextual, and graph-based representations.

The focus of this project is **not binary hate classification**, but understanding how hate
*emerges and intensifies over the course of a conversation*.

This codebase was developed as part of an academic project and primarily serves as a
**research and learning-oriented implementation**, rather than a polished production system.


# What this project tries to do

Given a tweet and its early replies, the goal is to model and forecast how hateful
the *future replies* in the conversation might become.

Instead of predicting weather the tweet is hateful or not it predicts the probability 
of conversation evolving into toxic content, it analyzes the conversation to detect early signs 
of hate

# High level idea
The model treats the conversation tweets as a time series data.
The pipeline broadly follows these steps:
1. Convert reply chains into windowed hate intensity sequences
2. Encode historical and future trends into latent representations
3. Group similar conversation trajectories using soft clustering
4. Use historical context + prior cluster information to predict future trends

