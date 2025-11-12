# Restaurant Recommendation System

A collaborative filtering-based restaurant recommendation system that predicts ratings for unrated restaurants based on user similarity.

## Overview

This system uses **cosine similarity** to find users with similar taste preferences and generates personalized restaurant recommendations with predicted ratings on a 0-10 scale.

## Features

- Collaborative filtering recommendation algorithm
- Cosine similarity-based user matching
- Weighted predictions based on similar users
- Top 10 restaurant recommendations
- Similarity analysis showing most similar users
- Confidence metrics for predictions

## Requirements

- Python 3.7 or higher
- pandas
- numpy
- scikit-learn

## Installation

1. **Clone or download this repository**

2. **Install required packages:**

```bash
pip install pandas numpy scikit-learn
```

Or using a requirements.txt file:

```bash
pip install -r requirements.txt
```

### requirements.txt
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
```

## File Structure

```
restaurant-recommender/
│
├── recommend.py              # Main recommendation script
├── restaurant_data.csv       # User ratings data
├── restaurant_features.csv   # Restaurant attributes (optional)
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Usage

1. **Ensure both CSV files are in the same directory as recommend.py**

2. **Run the script:**

```bash
python recommend.py
```

3. **Output includes:**
   - Complete ranked list of all unrated restaurants
   - Top 10 restaurant recommendations with predicted ratings
   - Top 10 most similar users based on taste preferences
   - Confidence analysis metrics

## How It Works

### 1. Similarity Calculation
The system uses **cosine similarity** to compare your ratings with other users:
- Only compares restaurants that both you and another user have rated
- Requires at least 3 common restaurants for valid comparison
- Similarity scores range from 0 (no correlation) to 1 (identical taste)

### 2. Rating Prediction
For each unrated restaurant:
- Collects ratings from all users who rated it
- Weights each rating by that user's similarity to you
- Calculates weighted average as predicted rating

### 3. Recommendation
Restaurants are ranked by predicted rating, with higher scores indicating better matches for your preferences.

## Data Format

### restaurant_data.csv
- First column: User identifier
- Remaining columns: Restaurant ratings (1-10 scale)
- Empty cells indicate unrated restaurants

Example:
```csv
User,Taco Bell,Chipotle,Subway,...
You,8,6,9,...
User1,9,8,2,...
```

### restaurant_features.csv (optional)
Contains restaurant metadata:
- Restaurant name
- Category (Chicken, Burger, Mexican, etc.)
- Price tier (1-3)
- Sauciness level (1-5)
- Popularity score (1-10)

*Note: Current version uses collaborative filtering only. Feature data not yet integrated.*

## Customization

### Modify Your Ratings
Edit the `your_ratings_dict` in `recommend.py`:

```python
your_ratings_dict = {
    'Taco Bell': 8,
    'Del Taco': 7,
    'Chipotle': 6,
    # Add more ratings...
}
```

### Adjust Similarity Threshold
Change minimum common restaurants required:

```python
if common_mask.sum() >= 3:  # Change 3 to your preferred minimum
```

### Change Number of Recommendations
Modify the display limit:

```python
for i, (restaurant, rating) in enumerate(sorted_predictions[:10], 1):  # Change 10 to desired number
```

## Algorithm Details

**Collaborative Filtering Approach:**
- Uses user-based collaborative filtering
- Cosine similarity metric for user comparison
- Weighted average for rating prediction
- Handles sparse data (missing ratings)

**Limitations:**
- Requires sufficient overlap between users' rated restaurants
- Doesn't consider restaurant features (category, price, etc.)
- Cold start problem for new users with few ratings

## Example Output

```
You have rated 28 restaurants.
Found 26 restaurants you haven't rated yet.

================================================================================
RESTAURANT RECOMMENDATIONS (Sorted by Predicted Rating)
================================================================================
Rank   Restaurant                                   Predicted Rating   
--------------------------------------------------------------------------------
1      Cheesecake Factory                           7.89/10
2      Olive Garden                                 7.45/10
3      Five Guys                                    7.23/10
...

TOP 10 RECOMMENDATIONS:
================================================================================
1. Cheesecake Factory                           Predicted: 7.89/10
2. Olive Garden                                 Predicted: 7.45/10
...

TOP 10 USERS WITH MOST SIMILAR TASTE:
================================================================================
1.  User35       Similarity: 0.892  (Based on 24 common restaurants)
2.  User12       Similarity: 0.867  (Based on 22 common restaurants)
...
```

## Troubleshooting

**Error: "No module named 'pandas'"**
- Solution: Install required packages using `pip install pandas numpy scikit-learn`

**Error: "FileNotFoundError: restaurant_data.csv"**
- Solution: Ensure CSV files are in the same directory as recommend.py

**Error: "ParserError: Error tokenizing data"**
- Solution: Check CSV file for formatting issues (extra commas, missing values)

**Low confidence predictions**
- Ensure enough users have rated the restaurants
- Check that you have sufficient ratings in common with other users
