import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the data with error handling
try:
    ratings_df = pd.read_csv('restaurant_data.csv', on_bad_lines='skip')
except:
    # Alternative: try with different parameters
    ratings_df = pd.read_csv('restaurant_data.csv', engine='python', on_bad_lines='skip')

print(f"Loaded data with {len(ratings_df)} users and {len(ratings_df.columns) - 1} restaurants\n")

# Get all columns (restaurants)
all_restaurants = ratings_df.columns[1:].tolist()

# Create your ratings dictionary based on the screenshots
your_ratings_dict = {
    'Taco Bell': 8, 'Del Taco': 7, 'Chipotle': 6, 'Subway': 9, 'Jersey Mikes': 8,
    'Mendocino Farms': 8, 'FireHouse Subs': 3, "Dave's Hot Chicken": 8, 'WingStop': 8,
    "Raisin' Canes": 7, 'In-N-Out': 8, "McDonald's": 7, 'Shake Shack': 6, "Carl's Jr.": 6,
    'Burger King': 6, 'Jack in the Box': 7, 'The Habit': 6, 'KFC': 7, "Popeyes": 9,
    'Chick-fil-A': 6, 'Sabrosada': 7, 'Subculture': 7, 'BB.Q Chicken': 7,
    'Bunz': 7, 'Street Taco Vendors': 7, "Ike's Love & Sandwiches": 7,
    'Jollibee': 6, 'El Pollo Loco': 7
}

# Create your ratings vector with NaN for unrated restaurants
your_vector = pd.Series({restaurant: your_ratings_dict.get(restaurant, np.nan)
                         for restaurant in all_restaurants})

# Get all other users' ratings (skip the "You" row if it exists)
other_users = ratings_df[ratings_df['User'] != 'You']
ratings_matrix = other_users.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
user_names = other_users['User'].values

# Find restaurants you haven't rated
unrated_restaurants = your_vector[your_vector.isna()].index.tolist()

print(f"You have rated {len(your_ratings_dict)} restaurants.")
print(f"Found {len(unrated_restaurants)} restaurants you haven't rated yet.\n")

# Calculate similarity between you and other users
similarities = []

for idx, user_row in ratings_matrix.iterrows():
    # Find common rated restaurants
    common_mask = ~your_vector.isna() & ~user_row.isna()

    if common_mask.sum() >= 3:  # Need at least 3 common restaurants
        your_common = your_vector[common_mask].values.reshape(1, -1)
        user_common = user_row[common_mask].values.reshape(1, -1)

        # Calculate cosine similarity
        sim = cosine_similarity(your_common, user_common)[0][0]
        similarities.append(sim)
    else:
        similarities.append(0)

similarities = np.array(similarities)

# Predict ratings for unrated restaurants
predictions = {}

for restaurant in unrated_restaurants:
    # Get ratings from other users for this restaurant
    restaurant_ratings = ratings_matrix[restaurant]

    # Filter out users who haven't rated this restaurant
    valid_mask = ~restaurant_ratings.isna()

    if valid_mask.sum() == 0:
        continue

    valid_ratings = restaurant_ratings[valid_mask].values
    valid_similarities = similarities[valid_mask]

    # Only consider users with positive similarity
    positive_sim_mask = valid_similarities > 0

    if positive_sim_mask.sum() > 0:
        # Weighted average based on similarity
        filtered_ratings = valid_ratings[positive_sim_mask]
        filtered_sims = valid_similarities[positive_sim_mask]
        predicted_rating = np.average(filtered_ratings, weights=filtered_sims)
    else:
        # Fall back to simple average if no similar users
        predicted_rating = valid_ratings.mean()

    predictions[restaurant] = predicted_rating

# Sort predictions by predicted rating
sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

# Display recommendations
print("=" * 80)
print("RESTAURANT RECOMMENDATIONS (Sorted by Predicted Rating)")
print("=" * 80)
print(f"{'Rank':<6} {'Restaurant':<45} {'Predicted Rating':<15}")
print("-" * 80)

for rank, (restaurant, rating) in enumerate(sorted_predictions, 1):
    print(f"{rank:<6} {restaurant:<45} {rating:.2f}/10")

print("\n" + "=" * 80)
print(f"TOP 10 RECOMMENDATIONS:")
print("=" * 80)
for i, (restaurant, rating) in enumerate(sorted_predictions[:10], 1):
    print(f"{i}. {restaurant:<45} Predicted: {rating:.2f}/10")

# Show which users are most similar to you
print("\n" + "=" * 80)
print("TOP 10 USERS WITH MOST SIMILAR TASTE:")
print("=" * 80)
top_similar_users = sorted(zip(user_names, similarities), key=lambda x: x[1], reverse=True)[:10]
for rank, (user, sim) in enumerate(top_similar_users, 1):
    # Count common restaurants
    user_idx = list(user_names).index(user)
    user_row = ratings_matrix.iloc[user_idx]
    common_count = (~your_vector.isna() & ~user_row.isna()).sum()
    print(f"{rank:<3}. {user:<12} Similarity: {sim:.3f}  (Based on {common_count} common restaurants)")

# Optional: Show confidence levels
print("\n" + "=" * 80)
print("CONFIDENCE ANALYSIS:")
print("=" * 80)
print(f"Average similarity score: {similarities[similarities > 0].mean():.3f}")
print(f"Users with positive similarity: {(similarities > 0).sum()}/{len(similarities)}")
print(f"\nNote: Predictions are more reliable for restaurants rated by similar users.")