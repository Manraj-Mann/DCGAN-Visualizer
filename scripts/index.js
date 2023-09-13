var eventSource;
// Chart initialization

var ctx = document.getElementById('loss-chart').getContext('2d');
var chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: 'Generator Loss',
        data: [],
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 2,
        fill: true,
        pointRadius: 1// Hide points (dots)
      },
      {
        label: 'Discriminator Loss',
        data: [],
        backgroundColor: 'rgba(255, 99, 132, 0.2)',
        borderColor: 'rgba(255, 99, 132, 1)',
        borderWidth: 2,
        fill: true,
        pointRadius: 1 // Hide points (dots)
      }
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        display: true,
        scaleLabel: {
          display: true,
          labelString: 'Epochs',
          font: {
            size: 14 // Adjust the font size as needed
          }
        },
        ticks: {
          autoSkip: true,
          maxTicksLimit: 10, // Maximum number of ticks to display
          stepSize: 2, // Fixed spacing between ticks
          callback: function (value, index, values) {
            // Customize the tick format as needed
            return value.toFixed(0); // Example: Show whole numbers
          },
          font: {
            size: 12 // Adjust the font size as needed
          }
        }
      },
      y: {
        display: true,
        scaleLabel: {

          display: true,
          labelString: 'Loss',
          font: {
            size: 14 // Adjust the font size as needed


          }
        },
        ticks: {
          font: {
            size: 12 // Adjust the font size as needed
          }
        }
      }
    }
  }
});

// Function to update the chart data
function updateChart(lossData) {
  // Append new data to the existing chart data

  // Calculate the starting epoch value based on the current chart data
  var startEpoch = chart.data.labels.length > 0 ? chart.data.labels[chart.data.labels.length - 1] + 1 : 1;

  // Generate continuous x-axis values starting from startEpoch
  var epochs = Array.from({ length: lossData.epochs.length }, (_, i) => startEpoch + i);

  // Append new data to the existing chart data
  chart.data.labels.push(...epochs);
  chart.data.datasets[0].data.push(...lossData.generatorLoss);
  chart.data.datasets[1].data.push(...lossData.discriminatorLoss);

  // Update the chart
  chart.update();
}

// Function to fetch data and update the chart
function fetchDataAndUpdateChart(url) {

  // Establish SSE connection to the server with query parameters

  eventSource = new EventSource(url);


  // Establish SSE connection to the server
  // eventSource = new EventSource('http://127.0.0.1:5000/get_data');

  // Event listener for receiving SSE stream data
  eventSource.addEventListener('message', function (event) {
    var lossData = JSON.parse(event.data);
    if (lossData.datafrom == "g") {

      set_generator(lossData);

    }
    else if (lossData.datafrom == "d") {

      set_discriminator(lossData);

    }
    else if (lossData.datafrom == "graph") {
      updateChart(lossData);
    }
  });

  // Error handling for SSE connection
  eventSource.addEventListener('error', function (event) {
    console.error('Error connecting to SSE stream:', event);
    eventSource.close();
  });
}

function set_discriminator(lossData) {


  var real_images_des = 'data:image/png;base64,' + lossData.real_images_des;
  var fake_images_des = 'data:image/png;base64,' + lossData.fake_images_des;

  var imageContainer1 = document.getElementById('real-images-div-img');
  var imageContainer2 = document.getElementById('fake-images-div-img');
  imageContainer1.src = real_images_des;
  imageContainer2.src = fake_images_des;

  set_latent(lossData);
  make_change_discriminator();
  BCE_loss(lossData.discriminatorLoss[0], " for Discriminator");


}

function set_generator(lossData) {

  var fake_images_gen = 'data:image/png;base64,' + lossData.fake_images_gen;
  var image = document.getElementById('fake-images-div-img');
  image.src = fake_images_gen;
  set_latent(lossData);
  make_change_generator();
  BCE_loss(lossData.generatorLoss[0], " for Generator");

}

function BCE_loss(loss, model) {

  var loss_fixed = loss.toFixed(4);
  // var loss = loss.toFixed(4);

  var loss_container = document.getElementById('loss-here');
  loss_container.innerHTML = loss_fixed + model;

}

function set_latent(lossData) {

  var latent = 'data:image/png;base64,' + lossData.latent;
  var image = document.getElementById('latent-chart');
  image.src = latent;

}

function make_change_generator() {

  var latent = document.getElementById('latent-chart');
  var generatorContainer = document.getElementById('generator-container');
  var generatorContainerImage = document.getElementById('fake-images-div-img');
  var descriminatorContainerImage = document.getElementById('real-images-div-img');
  var discriminatorContainer = document.getElementById('discriminator-container');
  var modelloss = document.getElementById('loss-container');
  var label = document.getElementById('training-cycle');
  var lablecontainer = document.querySelector('.bottom-right-label');
  
  descriminatorContainerImage.src = "../assets/No_image_available.svg.web.webp";
  latent.style.boxShadow = '0 0 2px 2px rgba(0, 255, 0, 0.5)';
  generatorContainer.style.boxShadow = '0 0 2px 2px rgba(0, 255, 0, 0.5)';
  generatorContainerImage.style.boxShadow = '0 0 2px 2px rgba(0, 255, 0, 0.5)';
  descriminatorContainerImage.style.boxShadow = "none";
  discriminatorContainer.style.boxShadow = "none";
  modelloss.style.boxShadow = '0 0 2px 2px rgba(0, 255, 0, 0.5)';
  label.innerHTML = "Training Cycle : Generator";
  lablecontainer.style.backgroundColor = "rgb(4, 184, 4)";
  
}
function make_change_discriminator() {
  
  var latent = document.getElementById('latent-chart');
  var generatorContainer = document.getElementById('generator-container');
  var generatorContainerImage = document.getElementById('fake-images-div-img');
  var descriminatorContainerImage = document.getElementById('real-images-div-img');
  var discriminatorContainer = document.getElementById('discriminator-container');
  var modelloss = document.getElementById('loss-container');
  var label = document.getElementById('training-cycle');
  var lablecontainer = document.querySelector('.bottom-right-label');
  
  generatorContainer.style.boxShadow = '0 0 2px 2px rgba(255, 0, 0, 0.5)';
  latent.style.boxShadow = '0 0 2px 2px rgba(255, 0, 0, 0.5)';
  generatorContainerImage.style.boxShadow = '0 0 2px 2px rgba(255, 0, 0, 0.5)';
  modelloss.style.boxShadow = '0 0 2px 2px rgba(255, 0, 0, 0.5)';
  descriminatorContainerImage.style.boxShadow = '0 0 2px 2px rgba(255, 0, 0, 0.5)';
  discriminatorContainer.style.boxShadow = '0 0 2px 2px rgba(255, 0, 0, 0.5)';
  label.innerHTML = "Training Cycle : Discriminator";
  lablecontainer.style.backgroundColor = "rgb(202, 15, 15)";

}


function play() {
  var pretrainedCheckbox = document.getElementById('pretrained-checkbox');
  var partiallyTrainedCheckbox = document.getElementById('partially-trained-checkbox');

  var pretrained = pretrainedCheckbox.checked ? 1 : 0;
  var partiallyTrained = partiallyTrainedCheckbox.checked ? 1 : 0;

  var url = 'http://127.0.0.1:5000/get_data?pretrained=' + pretrained + '&partially_trained=' + partiallyTrained;

  // Call the function to fetch data and update the chart
  fetchDataAndUpdateChart(url);
}

function stop() {
  // Close the SSE connection
  if (eventSource) {
    eventSource.close();
    eventSource = null;
  }
  location.reload();
}

set_generator_image();
set_discriminator_image();

function set_generator_image() {


  // Get a reference to the canvas element
  var canvas = document.getElementById('generator-chart');

  // Get the 2D rendering context of the canvas
  var ctx = canvas.getContext('2d');

  // Create a new Image object
  var image = new Image();
  // Set the src attribute of the Image object to the URL of the image
  image.src = '../assets/generator.jpg';

  // Add an event listener to the Image object's onload event
  // Add an event listener to the Image object's onload event
  image.onload = function () {
    // Calculate the scale to fit the image within the canvas
    var scale = Math.min(canvas.width / image.width, canvas.height / image.height);

    // Calculate the new width and height to maintain the aspect ratio
    var width = image.width * scale + 10;
    var height = image.height * scale + 10;

    // Calculate the position to center the image on the canvas
    var x = (canvas.width - width) / 2;
    var y = (canvas.height - height) / 2;

    // Draw the image onto the canvas with the calculated dimensions and position
    ctx.drawImage(image, x, y, width, height);
  };
}

function set_discriminator_image() {

  // Get a reference to the canvas element
  var canvas = document.getElementById('discriminator-chart');

  // Get the 2D rendering context of the canvas
  var ctx = canvas.getContext('2d');

  // Create a new Image object
  var image = new Image();
  // Set the src attribute of the Image object to the URL of the image
  image.src = '../assets/discriminator.jpg';

  // Add an event listener to the Image object's onload event
  // Add an event listener to the Image object's onload event
  image.onload = function () {
    // Calculate the scale to fit the image within the canvas
    var scale = Math.min(canvas.width / image.width, canvas.height / image.height);

    // Calculate the new width and height to maintain the aspect ratio
    var width = image.width * scale + 10;
    var height = image.height * scale + 10;

    // Calculate the position to center the image on the canvas
    var x = (canvas.width - width) / 2;
    var y = (canvas.height - height) / 2;

    // Draw the image onto the canvas with the calculated dimensions and position
    ctx.drawImage(image, x, y, width, height);
  };
}