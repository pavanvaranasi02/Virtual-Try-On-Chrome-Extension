chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "runFlaskApp") {
      // Replace 'http://yourflaskapp.com/upload' with your Flask app's endpoint
      fetch('http://127.0.0.1:8000', {
        method: 'POST',
        body: JSON.stringify({}),  // Include any data you want to send
        headers: {
          'Content-Type': 'application/json',
        },
      })
      .then(response => {
        if (response.ok) {
          console.log('Flask app request successful');
          // Handle the response here, if needed
        } else {
          console.error('Error while contacting Flask app:', response.status, response.statusText);
          // Handle the error here
        }
      })
      .catch(error => {
        console.error('Error during fetch:', error);
        // Handle network or fetch-related errors here
      });
    }
  });
  