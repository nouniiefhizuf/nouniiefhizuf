#!/usr/bin/env python3

import os
import argparse
import json
import random
from glob import glob
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def _find_first_csv(data_dir: str, name_patterns: List[str]) -> Optional[str]:
	if not data_dir or not os.path.isdir(data_dir):
		return None
	all_csvs = glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
	candidates = []
	for path in all_csvs:
		lower = os.path.basename(path).lower()
		for pat in name_patterns:
			if pat in lower:
				candidates.append(path)
				break
	if candidates:
		candidates.sort(key=lambda p: (p.count(os.sep), len(os.path.basename(p))))
		return candidates[0]
	return None


def load_ratings_df(data_dir: str) -> pd.DataFrame:
	path = _find_first_csv(data_dir, ["rating", "ratings"]) if data_dir else None
	if path is None:
		raise FileNotFoundError("Could not find ratings CSV. Ensure a file like 'ratings.csv' exists.")
	df = pd.read_csv(path)
	colmap = {c.lower(): c for c in df.columns}
	user_col = next((colmap[k] for k in colmap if k in ["userid", "user_id", "user"]), None)
	item_col = next((colmap[k] for k in colmap if k in ["movieid", "movie_id", "movie", "itemid", "item_id", "item"]), None)
	rating_col = next((colmap[k] for k in colmap if k in ["rating", "score", "rank", "stars"]), None)
	ts_col = next((colmap[k] for k in colmap if k in ["timestamp", "ts", "time"]), None)
	if not (user_col and item_col and rating_col):
		raise ValueError(f"Ratings file at {path} is missing required columns. Found: {list(df.columns)}")
	df = df.rename(columns={user_col: "user_id", item_col: "movie_id", rating_col: "rating"})
	if ts_col:
		df = df.rename(columns={ts_col: "timestamp"})
	df = df.dropna(subset=["user_id", "movie_id", "rating"]).copy()
	if not np.issubdtype(df["user_id"].dtype, np.number):
		df["user_id"], _ = pd.factorize(df["user_id"])  # start at 0
		df["user_id"] = df["user_id"].astype(int)
	if not np.issubdtype(df["movie_id"].dtype, np.number):
		df["movie_id"], _ = pd.factorize(df["movie_id"])  # start at 0
		df["movie_id"] = df["movie_id"].astype(int)
	df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
	df = df.dropna(subset=["rating"])  # remove invalid ratings
	rating_min, rating_max = df["rating"].min(), df["rating"].max()
	if rating_min < 0 or rating_max > 10:
		df["rating"] = df["rating"].clip(lower=0, upper=10)
	if df.duplicated(subset=["user_id", "movie_id"]).any():
		df = df.groupby(["user_id", "movie_id"], as_index=False).agg({"rating": "mean", "timestamp": "max" if "timestamp" in df.columns else "first"})
	return df


def load_movies_df(data_dir: str) -> pd.DataFrame:
	path = _find_first_csv(data_dir, ["movie", "movies", "title", "titles"]) or _find_first_csv(data_dir, ["metadata", "tmdb", "films", "items"])
	if path is None:
		raise FileNotFoundError("Could not find movies metadata CSV. Ensure a file like 'movies.csv' exists.")
	df = pd.read_csv(path)
	colmap = {c.lower(): c for c in df.columns}
	id_col = next((colmap[k] for k in colmap if k in ["movieid", "movie_id", "movie", "itemid", "item_id", "id"]), None)
	title_col = next((colmap[k] for k in colmap if k in ["title", "name"]), None)
	genres_col = next((colmap[k] for k in colmap if k in ["genres", "genre", "listed_in"]), None)
	overview_col = next((colmap[k] for k in colmap if k in ["overview", "description", "plot", "summary"]), None)
	keywords_col = next((colmap[k] for k in colmap if k in ["keywords", "tags"]), None)
	if id_col is None:
		if title_col is None:
			raise ValueError(f"Movies metadata at {path} lacks identifiable id or title columns: {list(df.columns)}")
		df["movie_id"], _ = pd.factorize(df[title_col])
	else:
		df = df.rename(columns={id_col: "movie_id"})
	if title_col:
		df = df.rename(columns={title_col: "title"})
	else:
		df["title"] = df["movie_id"].astype(str)
	if genres_col:
		df = df.rename(columns={genres_col: "genres"})
	else:
		df["genres"] = None
	if overview_col:
		df = df.rename(columns={overview_col: "overview"})
	else:
		df["overview"] = None
	if keywords_col:
		df = df.rename(columns={keywords_col: "keywords"})
	else:
		df["keywords"] = None
	if not np.issubdtype(df["movie_id"].dtype, np.number):
		df["movie_id"], _ = pd.factorize(df["movie_id"])  # start at 0
		df["movie_id"] = df["movie_id"].astype(int)
	def _norm_text(x):
		if pd.isna(x):
			return ""
		if isinstance(x, str):
			return x.replace("|", ",").replace("/", ",").replace(";", ",").lower()
		return str(x)
	df["genres"] = df["genres"].apply(_norm_text)
	df["overview"] = df["overview"].apply(_norm_text)
	df["keywords"] = df["keywords"].apply(_norm_text)
	df = df.drop_duplicates(subset=["movie_id"]).copy()
	return df[["movie_id", "title", "genres", "overview", "keywords"]]


def summarize_dataset(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
	num_users = ratings["user_id"].nunique()
	num_movies_rated = ratings["movie_id"].nunique()
	num_movies_meta = movies["movie_id"].nunique()
	num_ratings = len(ratings)
	sparsity = 1.0 - (num_ratings / (num_users * max(1, num_movies_rated)))
	rating_min, rating_max = ratings["rating"].min(), ratings["rating"].max()
	print(f"Users: {num_users:,} | Movies (rated): {num_movies_rated:,} | Movies (meta): {num_movies_meta:,} | Ratings: {num_ratings:,}")
	print(f"Rating scale observed: [{rating_min}, {rating_max}] | Sparsity: {sparsity:.4f}")
	return pd.DataFrame({
		"num_users": [num_users],
		"num_movies_rated": [num_movies_rated],
		"num_movies_meta": [num_movies_meta],
		"num_ratings": [num_ratings],
		"sparsity": [sparsity],
		"rating_min": [rating_min],
		"rating_max": [rating_max],
	})


def plot_eda(ratings: pd.DataFrame, movies: pd.DataFrame, outdir: Optional[str] = None) -> None:
	plt.figure(figsize=(6,4))
	sns.histplot(ratings["rating"], bins=20, kde=True)
	plt.title("Rating Distribution")
	plt.tight_layout()
	if outdir:
		plt.savefig(os.path.join(outdir, "eda_rating_distribution.png"))
	plt.show()

	movie_counts = ratings.groupby("movie_id").size().sort_values(ascending=False).head(20)
	mc_df = movie_counts.reset_index(name="num_ratings").merge(movies[["movie_id", "title"]], on="movie_id", how="left")
	plt.figure(figsize=(8,6))
	sns.barplot(data=mc_df, y="title", x="num_ratings")
	plt.title("Top 20 Most Rated Movies")
	plt.tight_layout()
	if outdir:
		plt.savefig(os.path.join(outdir, "eda_top_movies.png"))
	plt.show()

	user_counts = ratings.groupby("user_id").size().sort_values(ascending=False).head(20).reset_index(name="num_ratings")
	plt.figure(figsize=(8,6))
	sns.barplot(data=user_counts, x="user_id", y="num_ratings")
	plt.title("Top 20 Most Active Users")
	plt.tight_layout()
	if outdir:
		plt.savefig(os.path.join(outdir, "eda_top_users.png"))
	plt.show()

	# Genre popularity
	genre_counts = {}
	for g in movies["genres"].fillna(""):
		for token in [t.strip() for t in g.split(',') if t.strip()]:
			genre_counts[token] = genre_counts.get(token, 0) + 1
	if genre_counts:
		g_df = pd.DataFrame(sorted(genre_counts.items(), key=lambda kv: kv[1], reverse=True)[:20], columns=["genre", "count"])
		plt.figure(figsize=(8,6))
		sns.barplot(data=g_df, y="genre", x="count")
		plt.title("Top Genres (by count in metadata)")
		plt.tight_layout()
		if outdir:
			plt.savefig(os.path.join(outdir, "eda_genres.png"))
		plt.show()


# Simple Matrix Factorization (SGD) for CF
class MF:
	def __init__(self, num_users: int, num_items: int, num_factors: int = 50, learning_rate: float = 0.01, reg: float = 0.02, n_epochs: int = 20, random_state: int = RANDOM_SEED):
		self.num_users = num_users
		self.num_items = num_items
		self.num_factors = num_factors
		self.learning_rate = learning_rate
		self.reg = reg
		self.n_epochs = n_epochs
		np.random.seed(random_state)
		self.global_mean = 0.0
		self.user_factors = np.random.normal(0, 0.1, (num_users, num_factors))
		self.item_factors = np.random.normal(0, 0.1, (num_items, num_factors))
		self.user_bias = np.zeros(num_users)
		self.item_bias = np.zeros(num_items)

	def fit(self, df: pd.DataFrame) -> None:
		self.global_mean = df["rating"].mean()
		for epoch in range(self.n_epochs):
			shuffled = df.sample(frac=1.0, random_state=RANDOM_SEED + epoch)
			for _, row in shuffled.iterrows():
				u = int(row["user_id"])
				i = int(row["movie_id"])
				r = float(row["rating"])
				pred = self.predict_single(u, i)
				err = r - pred
				# Update biases
				self.user_bias[u] += self.learning_rate * (err - self.reg * self.user_bias[u])
				self.item_bias[i] += self.learning_rate * (err - self.reg * self.item_bias[i])
				# Update latent factors
				pu = self.user_factors[u]
				qi = self.item_factors[i]
				self.user_factors[u] += self.learning_rate * (err * qi - self.reg * pu)
				self.item_factors[i] += self.learning_rate * (err * pu - self.reg * qi)

	def predict_single(self, user_id: int, item_id: int) -> float:
		pred = self.global_mean + self.user_bias[user_id] + self.item_bias[item_id] + np.dot(self.user_factors[user_id], self.item_factors[item_id])
		return float(pred)

	def predict_many(self, user_id: int, item_ids: List[int]) -> np.ndarray:
		ub = self.user_bias[user_id]
		uf = self.user_factors[user_id]
		preds = self.global_mean + ub + self.item_bias[item_ids] + uf @ self.item_factors[item_ids].T
		return preds


def train_mf_cf(ratings: pd.DataFrame) -> Tuple[MF, float]:
	num_users = int(ratings["user_id"].max()) + 1
	num_items = int(ratings["movie_id"].max()) + 1
	mf = MF(num_users, num_items, num_factors=32, learning_rate=0.01, reg=0.05, n_epochs=25, random_state=RANDOM_SEED)
	# Train/test split
	mask = np.random.rand(len(ratings)) < 0.8
	train_df = ratings[mask].copy()
	test_df = ratings[~mask].copy()
	mf.fit(train_df)
	# Evaluate RMSE
	preds = []
	for _, row in test_df.iterrows():
		u = int(row["user_id"]) 
		i = int(row["movie_id"]) 
		r = float(row["rating"]) 
		preds.append((r - mf.predict_single(u, i)) ** 2)
	rmse = float(np.sqrt(np.mean(preds))) if preds else float("nan")
	print(f"MF (SGD) -> RMSE: {rmse:.4f}")
	return mf, rmse


def build_content_similarity(movies: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
	content_series = (movies["title"].fillna("") + " " + movies["genres"].fillna("") + " " + movies["keywords"].fillna("") + " " + movies["overview"].fillna("")).astype(str)
	vectorizer = TfidfVectorizer(stop_words="english", min_df=1)
	matrix = vectorizer.fit_transform(content_series)
	sim_matrix = linear_kernel(matrix, matrix)
	return movies[["movie_id", "title"]].reset_index(drop=True), sim_matrix


def recommend_cbf_for_movie(movie_id: int, movies_df: pd.DataFrame, sim_matrix: np.ndarray, top_k: int = 10) -> List[int]:
	movie_index_map = {mid: idx for idx, mid in enumerate(movies_df["movie_id"]) }
	if movie_id not in movie_index_map:
		return []
	idx = movie_index_map[movie_id]
	sim_scores = list(enumerate(sim_matrix[idx]))
	sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
	top_indices = [i for i, _ in sim_scores[1: top_k+1]]
	return movies_df.iloc[top_indices]["movie_id"].tolist()


def recommend_cf_for_user(user_id: int, mf: MF, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
	seen = set(ratings_df.loc[ratings_df["user_id"] == user_id, "movie_id"].tolist())
	candidates = [mid for mid in movies_df["movie_id"].tolist() if mid not in seen]
	candidates_clipped = [mid for mid in candidates if mid < mf.num_items]
	if not candidates_clipped:
		return pd.DataFrame(columns=["movie_id", "pred_rating", "title"])
	pred_values = [ (mid, mf.predict_single(user_id, mid)) for mid in candidates_clipped ]
	pred_values.sort(key=lambda x: x[1], reverse=True)
	top = pred_values[:n]
	res = pd.DataFrame(top, columns=["movie_id", "pred_rating"]).merge(movies_df[["movie_id", "title"]], on="movie_id", how="left")
	return res


def recommend_hybrid(user_id: int, mf: MF, movies_df: pd.DataFrame, ratings_df: pd.DataFrame, sim_matrix: np.ndarray, alpha: float = 0.7, n: int = 10) -> pd.DataFrame:
	seen = set(ratings_df.loc[ratings_df["user_id"] == user_id, "movie_id"].tolist())
	candidates = [mid for mid in movies_df["movie_id"].tolist() if mid not in seen and mid < mf.num_items]
	# CF scores
	cf_scores = { mid: mf.predict_single(user_id, mid) for mid in candidates }
	# CBF from liked movies
	user_hist = ratings_df[ratings_df["user_id"] == user_id]
	if user_hist.empty:
		return recommend_cf_for_user(user_id, mf, movies_df, ratings_df, n)
	liked = user_hist[user_hist["rating"] >= user_hist["rating"].mean()]["movie_id"].tolist() or user_hist.sort_values("rating", ascending=False)["movie_id"].head(5).tolist()
	movie_index_map = {mid: idx for idx, mid in enumerate(movies_df["movie_id"]) }
	cbf_scores = { mid: 0.0 for mid in candidates }
	for lm in liked:
		if lm not in movie_index_map:
			continue
		lidx = movie_index_map[lm]
		sims = sim_matrix[lidx]
		for mid in candidates:
			idx = movie_index_map.get(mid)
			if idx is not None:
				cbf_scores[mid] += sims[idx]
	# Normalize CBF to 0..1
	vals = np.array(list(cbf_scores.values()))
	if vals.size > 0 and vals.max() > 0:
		for k in cbf_scores:
			cbf_scores[k] = cbf_scores[k] / vals.max()
	# Blend
	scores = { mid: alpha * cf_scores.get(mid, 0.0) + (1 - alpha) * cbf_scores.get(mid, 0.0) for mid in candidates }
	top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:n]
	res = pd.DataFrame(top, columns=["movie_id", "score"]).merge(movies_df[["movie_id", "title"]], on="movie_id", how="left")
	return res


def build_recommender(data_dir: str):
	print("Loading data from:", data_dir)
	ratings_df = load_ratings_df(data_dir)
	movies_df = load_movies_df(data_dir)
	summary = summarize_dataset(ratings_df, movies_df)
	print(summary.to_string(index=False))
	print("Running EDA plots...")
	plot_eda(ratings_df, movies_df, outdir=data_dir)
	print("Training MF CF model...")
	mf, rmse = train_mf_cf(ratings_df)
	print("Building content similarity matrix...")
	movies_small, sim_matrix = build_content_similarity(movies_df)
	return ratings_df, movies_df, mf, movies_small, sim_matrix


def recommend_movies(user_id: int, n: int, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, mf: MF, movies_small: pd.DataFrame, sim_matrix: np.ndarray, strategy: str = "hybrid", alpha: float = 0.7) -> pd.DataFrame:
	if strategy == "cf":
		return recommend_cf_for_user(user_id, mf, movies_df, ratings_df, n)
	elif strategy == "cbf":
		# Recommend similar to user's best-rated movie
		hist = ratings_df[ratings_df["user_id"] == user_id]
		if hist.empty:
			return pd.DataFrame(columns=["movie_id", "title"])  # cold-start
		best_movie = hist.sort_values("rating", ascending=False)["movie_id"].iloc[0]
		rec_ids = recommend_cbf_for_movie(best_movie, movies_small, sim_matrix, top_k=n)
		return movies_df[movies_df["movie_id"].isin(rec_ids)][["movie_id", "title"]]
	else:
		return recommend_hybrid(user_id, mf, movies_df, ratings_df, sim_matrix, alpha=alpha, n=n)


def add_new_ratings(ratings_df: pd.DataFrame, new_ratings: pd.DataFrame) -> pd.DataFrame:
	cols = {c.lower(): c for c in new_ratings.columns}
	user = next((cols[k] for k in cols if k in ["user_id", "userid", "user"]), None)
	movie = next((cols[k] for k in cols if k in ["movie_id", "movieid", "movie"]), None)
	rating = next((cols[k] for k in cols if k in ["rating", "score", "stars"]), None)
	assert user and movie and rating, "new_ratings must have user_id, movie_id, rating"
	df = new_ratings.rename(columns={user: "user_id", movie: "movie_id", rating: "rating"})[["user_id", "movie_id", "rating"]].copy()
	return pd.concat([ratings_df[["user_id", "movie_id", "rating"]], df], ignore_index=True)


def main():
	parser = argparse.ArgumentParser(description="Movie Recommender (CF + CBF + Hybrid)")
	parser.add_argument("--data_dir", type=str, required=True, help="Directory containing CSV files")
	parser.add_argument("--user_id", type=int, default=0, help="User ID to recommend for")
	parser.add_argument("--n", type=int, default=10, help="Number of recommendations")
	parser.add_argument("--strategy", type=str, default="hybrid", choices=["cf", "cbf", "hybrid"], help="Recommendation strategy")
	parser.add_argument("--alpha", type=float, default=0.7, help="Hybrid blend alpha (CF weight)")
	args = parser.parse_args()

	ratings_df, movies_df, mf, movies_small, sim_matrix = build_recommender(args.data_dir)
	print(f"\nTop {args.n} recommendations for user {args.user_id} using {args.strategy}:")
	recs = recommend_movies(args.user_id, args.n, ratings_df, movies_df, mf, movies_small, sim_matrix, strategy=args.strategy, alpha=args.alpha)
	print(recs.to_string(index=False))


if __name__ == "__main__":
	main()