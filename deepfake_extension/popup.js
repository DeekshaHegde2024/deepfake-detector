document.getElementById('check-btn').addEventListener('click', () => {
  const button = document.getElementById('check-btn');
  const results = document.getElementById('results');

  // Disable button and show loading state
  button.disabled = true;
  results.innerHTML = "<p> Analyzing the video... Please wait.</p>";

  chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
    const url = tabs[0].url;

    // Ensure it's a YouTube video URL
    if (!url.includes("youtube.com") && !url.includes("youtu.be")) {
      results.innerHTML = "<p> Please open a valid YouTube video.</p>";
      button.disabled = false;
      return;
    }

    fetch(`http://localhost:8000/analyze?url=${encodeURIComponent(url)}`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Server error: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        const format = num => Number(num).toFixed(3);

        results.innerHTML = `
          <p><strong> Media Score:</strong> ${format(data.media_score)}</p>
          <p><strong> Graph Score:</strong> ${format(data.graph_score)}</p>
          <p><strong> Sentiment Score:</strong> ${format(data.sentiment_score)}</p>
          <p><strong> Bot Flag:</strong> ${data.bot_flag ? 'True' : 'False'}</p>
        `;
      })
      .catch(err => {
        results.innerHTML = `<p> Error: ${err.message}</p>`;
      })
      .finally(() => {
        button.disabled = false;
      });
  });
});
