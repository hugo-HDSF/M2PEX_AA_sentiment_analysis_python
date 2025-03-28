document.addEventListener('DOMContentLoaded', function () {
  // Tweet Analyzer
  const analyzeBtn = document.getElementById('analyzeBtn');
  if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
      const tweetInput = document.getElementById('tweetInput');
      const tweets = tweetInput.value.trim().split('\n').filter(tweet => tweet.trim() !== '');
      
      if (tweets.length === 0) {
        alert('Please enter at least one tweet');
        return;
      }
      
      try {
        // Show loading state
        analyzeBtn.disabled = true;
        analyzeBtn.innerText = 'Analyzing...';
        
        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({tweets}),
        });
        
        const results = await response.json();
        
        if (response.ok) {
          displayResults(results);
        } else {
          alert('Error: ' + (results.error || 'Unknown error'));
        }
      } catch (error) {
        console.error('Error:', error);
        alert('An error occurred. Please try again later.');
      } finally {
        // Reset button state
        analyzeBtn.disabled = false;
        analyzeBtn.innerText = 'Analyze';
      }
    });
    
    function displayResults(results) {
      const resultsContainer = document.getElementById('results');
      const resultsList = document.getElementById('resultsList');
      
      resultsList.innerHTML = '';
      resultsContainer.classList.remove('hidden');
      
      for (const [tweet, score] of Object.entries(results)) {
        const resultItem = document.createElement('div');
        resultItem.className = 'result-item';
        
        const sentimentClass = score > 0.2 ? 'positive' :
          score < -0.2 ? 'negative' :
            'neutral';
        
        const sentimentText = score > 0.2 ? 'Positive' :
          score < -0.2 ? 'Negative' :
            'Neutral';
        
        resultItem.innerHTML = `
                    <div class="tweet">${tweet}</div>
                    <div class="score ${sentimentClass}">
                        <span class="sentiment">${sentimentText}</span>
                        <span class="value">${score.toFixed(2)}</span>
                    </div>
                `;
        
        resultsList.appendChild(resultItem);
      }
    }
  }
});