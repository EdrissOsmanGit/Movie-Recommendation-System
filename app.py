from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import os

app = Flask(__name__)

print("Loading data and building recommenders...")
ratings = pd.read_parquet("processed/ratings.parquet")
splits  = pd.read_parquet("processed/splits.parquet")
movies  = pd.read_parquet("processed/movies_enriched.parquet")

from pipeline_recommender import PipelineRecommender
rec = PipelineRecommender.build(ratings, splits, movies)
movie_lookup = movies.set_index("movieId")
print("Ready — open http://localhost:5000")

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Movie Recommender</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, sans-serif; background: #f5f5f5;
           display: flex; justify-content: center; padding: 2rem 1rem; }
    .container { width: 100%; max-width: 720px; }
    h1 { font-size: 1.6rem; margin-bottom: 0.25rem; }
    p.sub { color: #666; margin-bottom: 1.5rem; font-size: 0.9rem; }
    section { background: white; border-radius: 12px; padding: 1.25rem;
              margin-bottom: 1.25rem; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }
    section h2 { font-size: 1rem; margin-bottom: 0.75rem; color: #333; }
    input[type=text], textarea {
      width: 100%; padding: 0.6rem 0.8rem; border: 1px solid #ddd;
      border-radius: 8px; font-size: 0.95rem; outline: none; font-family: inherit; }
    input[type=text]:focus, textarea:focus { border-color: #6366f1; }
    textarea { resize: vertical; min-height: 80px; }
    .search-results { margin-top: 0.5rem; }
    .search-item { display: flex; justify-content: space-between; align-items: center;
                   padding: 0.5rem 0.75rem; border-radius: 8px; font-size: 0.9rem;
                   border: 1px solid #eee; margin-bottom: 4px; background: white; }
    .add-btn { background: #6366f1; color: white; border: none;
               border-radius: 6px; padding: 0.3rem 0.7rem;
               font-size: 0.85rem; cursor: pointer; white-space: nowrap; }
    .add-btn:hover { background: #4f46e5; }
    .history-chips { display: flex; flex-wrap: wrap; gap: 0.4rem;
                     min-height: 32px; margin-top: 0.75rem; }
    .chip { background: #ede9fe; color: #4338ca; border-radius: 999px;
            padding: 0.25rem 0.75rem; font-size: 0.82rem; display: flex;
            align-items: center; gap: 0.4rem; }
    .chip button { background: none; border: none; cursor: pointer;
                   color: #6366f1; font-size: 1.1rem; line-height: 1; padding: 0; }
    .ask-btn { margin-top: 0.75rem; width: 100%; padding: 0.7rem;
               background: #6366f1; color: white; border: none;
               border-radius: 8px; font-size: 1rem; cursor: pointer; }
    .ask-btn:hover { background: #4f46e5; }
    .ask-btn:disabled { background: #a5b4fc; cursor: not-allowed; }
    .results-list { display: flex; flex-direction: column; gap: 0.75rem; }
    .movie-card { border: 1px solid #eee; border-radius: 10px; padding: 0.9rem 1rem; }
    .movie-card .rank { font-size: 0.75rem; color: #999; margin-bottom: 0.2rem; }
    .movie-card .title { font-weight: 600; font-size: 1rem; }
    .movie-card .genres { font-size: 0.8rem; color: #6366f1; margin-top: 0.2rem; }
    .movie-card .overview { font-size: 0.85rem; color: #555; margin-top: 0.4rem; line-height: 1.5; }
    .empty { color: #999; font-size: 0.9rem; text-align: center; padding: 1rem 0; }
    .spinner { text-align: center; color: #6366f1; padding: 1rem; }
    .error { color: #dc2626; font-size: 0.9rem; text-align: center; padding: 1rem 0; }
  </style>
</head>
<body>
<div class="container">
  <h1>Movie Recommender</h1>
  <p class="sub">Tell us what you're in the mood for and we'll find something great.</p>

  <section>
    <h2>1. Add movies you've enjoyed</h2>
    <input type="text" id="search-input" placeholder="Search for a movie title...">
    <div class="search-results" id="search-results"></div>
    <div class="history-chips" id="history-chips">
      <span class="empty" id="history-empty">No movies added yet</span>
    </div>
  </section>

  <section>
    <h2>2. Describe what you want to watch</h2>
    <textarea id="query-input"
      placeholder="e.g. Something like Inception but lighter, a sci-fi thriller from the last 10 years, nothing too violent"></textarea>
    <button class="ask-btn" id="ask-btn">Get recommendations</button>
  </section>

  <section id="results-section" style="display:none">
    <h2>Recommendations</h2>
    <div id="results-list" class="results-list"></div>
  </section>
</div>

<script>
  var likedMovies = {};

  document.getElementById("search-input").addEventListener("input", function() {
    var q = this.value;
    clearTimeout(window._searchTimer);
    if (q.length < 2) {
      document.getElementById("search-results").innerHTML = "";
      return;
    }
    window._searchTimer = setTimeout(function() {
      fetch("/search?q=" + encodeURIComponent(q))
        .then(function(r) { return r.json(); })
        .then(function(hits) {
          var el = document.getElementById("search-results");
          if (!hits.length) {
            el.innerHTML = '<div class="empty">No results found</div>';
            return;
          }
          var html = "";
          for (var i = 0; i < hits.length; i++) {
            var m = hits[i];
            var label = m.title + (m.year ? " (" + m.year + ")" : "");
            html += '<div class="search-item">'
                  + '<span>' + label + '</span>'
                  + '<button class="add-btn" data-id="' + m.movieId + '" data-title="' + label + '">+ Add</button>'
                  + '</div>';
          }
          el.innerHTML = html;
          el.querySelectorAll(".add-btn").forEach(function(btn) {
            btn.addEventListener("click", function() {
              addMovie(parseInt(this.dataset.id), this.dataset.title);
            });
          });
        })
        .catch(function(e) { console.error("Search error:", e); });
    }, 300);
  });

  function addMovie(id, title) {
    if (likedMovies[id]) return;
    likedMovies[id] = title;
    renderChips();
    document.getElementById("search-results").innerHTML = "";
    document.getElementById("search-input").value = "";
  }

  function removeMovie(id) {
    delete likedMovies[id];
    renderChips();
  }

  function renderChips() {
    var el = document.getElementById("history-chips");
    var keys = Object.keys(likedMovies);
    var emptyEl = document.getElementById("history-empty");
    if (keys.length === 0) {
      el.innerHTML = '<span class="empty" id="history-empty">No movies added yet</span>';
      return;
    }
    var html = "";
    for (var i = 0; i < keys.length; i++) {
      var id = keys[i];
      html += '<span class="chip">' + likedMovies[id]
            + ' <button onclick="removeMovie(' + id + ')" title="Remove">×</button>'
            + '</span>';
    }
    el.innerHTML = html;
  }

  document.getElementById("ask-btn").addEventListener("click", function() {
    var query = document.getElementById("query-input").value.trim();
    if (!query) { alert("Please describe what you want to watch."); return; }

    var historyIds = Object.keys(likedMovies).map(Number);
    var btn = document.getElementById("ask-btn");
    btn.disabled = true;
    btn.textContent = "Thinking... (this can take 30-60s)";

    var section = document.getElementById("results-section");
    var list = document.getElementById("results-list");
    section.style.display = "block";
    list.innerHTML = '<div class="spinner">Fetching candidates and reranking with LLM...</div>';

    fetch("/recommend", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, history_ids: historyIds, k: 10 })
    })
    .then(function(r) {
      if (!r.ok) throw new Error("Server error: " + r.status);
      return r.json();
    })
    .then(function(data) {
      var recs = data.recommendations;
      if (!recs || !recs.length) {
        list.innerHTML = '<div class="empty">No results found. Try a different query.</div>';
        return;
      }
      var html = "";
      for (var i = 0; i < recs.length; i++) {
        var m = recs[i];
        html += '<div class="movie-card">'
              + '<div class="rank">#' + (i+1) + '</div>'
              + '<div class="title">' + m.title + (m.year ? ' (' + m.year + ')' : '') + '</div>'
              + '<div class="genres">' + (m.genres || []).join(" · ") + '</div>'
              + (m.overview ? '<div class="overview">' + m.overview + '...</div>' : '')
              + '</div>';
      }
      list.innerHTML = html;
    })
    .catch(function(e) {
      console.error("Recommend error:", e);
      list.innerHTML = '<div class="error">Something went wrong: ' + e.message
                     + '<br>Check that <code>ollama serve</code> is running in a terminal.</div>';
    })
    .finally(function() {
      btn.disabled = false;
      btn.textContent = "Get recommendations";
    });
  });
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return HTML

@app.route("/recommend", methods=["POST"])
def recommend():
    data        = request.json
    query       = data.get("query", "")
    history_ids = data.get("history_ids", [])
    k           = data.get("k", 10)

    try:
        ranked_ids = rec.recommend(query, history_ids, k=k)
    except Exception as e:
        import traceback
        traceback.print_exc()   # prints full error to terminal
        return jsonify({"error": str(e)}), 500

    results = []
    for mid in ranked_ids:
        if mid not in movie_lookup.index:
            continue
        row = movie_lookup.loc[mid]
        genres = row.get("genres")
        results.append({
            "movieId":  int(mid),
            "title":    str(row["title"]),
            "year":     int(row["year"]) if pd.notna(row.get("year")) else None,
            "genres":   list(genres) if genres is not None else [],
            "overview": str(row.get("overview") or "")[:200],
        })
    return jsonify({"recommendations": results})

@app.route("/search", methods=["GET"])
def search():
    q = request.args.get("q", "").lower().strip()
    if len(q) < 2:
        return jsonify([])
    mask = movies["title"].str.lower().str.contains(q, na=False)
    hits = movies[mask].head(8)
    out  = []
    for r in hits.itertuples():
        out.append({
            "movieId": int(r.movieId),
            "title":   str(r.title),
            "year":    int(r.year) if pd.notna(r.year) else None,
        })
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=False, port=5000)