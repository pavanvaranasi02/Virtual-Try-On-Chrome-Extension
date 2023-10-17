document.addEventListener('DOMContentLoaded', function() {
  const myButton = document.getElementById('myButton');

  myButton.addEventListener('click', function(event) {
    event.preventDefault();
    performTask();
  });


  function performTask() {
    // Get the form data.
    console.log("Hello World..");
    const myForm = document.getElementById('myForm');
    const progressBar = document.getElementById("progressBar");
    progressBar.innerText = "Loading.... Please Wait for 40 secs."
    progressBar.style.color = "red";
    const formData = new FormData(myForm);

    // Get the files from the form data.
    const personImage = formData.get('personImage');
    const clothImage = formData.get('clothImage');

    // Check if the files are valid.
    if (!personImage || !clothImage) {
      // No files were selected.
      alert('Please select both person and cloth images.');
      return;
    }

    // if (personImage.width !== 192 || personImage.height !== 256 || clothImage.width !== 192 || clothImage.height !== 256) {
    //   // Images are not of the correct size.
    //   alert('Please select images of size 192*256.');
    //   return;
    // }


    // Post the form data to the Flask function.
    try {
      // Add a .then() method here
      fetch('http://127.0.0.1:8000/upload1', 
      {   method: 'POST',   
          mode: 'cors',   
          body: formData,
          insecure: true
      }).then(response => {
        // Check if the response status is 200
        if (response.status === 200) {
          // Parse the JSON response and return another promise
          return response.json();
        } else {
          // The function did not return a result.
          // Handle the error.
          console.log('no response.')
          throw new Error('Response status is not 200');
        }
      }).then(responseData => {
        // Access the parsed JSON data
        console.log(responseData);
        const resultImageUrl = responseData.result_image_url;
        const inputImageUrl = responseData.input_image_url;
        console.log(resultImageUrl);
        console.log(inputImageUrl);

        const time = new Date().getTime();

        // Create an image element and set its src attribute to the URL of the result image.
        const resultImage = document.createElement('img');
        resultImage.src = 'file:///'+resultImageUrl + '?timestamp=' + time;

        // Append the image element to the result-image div.
        const resultImageDiv = document.getElementById('result-image');
        resultImageDiv.appendChild(resultImage);

        // Create an image element and set its src attribute to the URL of the result image.
        const inputImage = document.createElement('img');
        inputImage.src = 'file:///'+inputImageUrl + '?timestamp=' + time;

        // Append the image element to the input-image div.
        const inputImageDiv = document.getElementById('input-image');
        inputImageDiv.appendChild(inputImage);



        // Display a success message to the user.
        alert('Your virtual try on is ready!');
        progressBar.innerHTML = "This is a test, Upload the image";
        progressBar.style.color = "blue";
      });
    } catch (error) {
      // Handle any errors that occur during the fetch or parsing.
      console.error("Error:", error);
      alert('Something went wrong. Please try again.');
      progressBar.innerHTML = "Sorry there is an error! Try Again...";
    }

  }
});