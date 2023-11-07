import React, { Component } from 'react';
import axios from 'axios';
import model1 from './Screenshot (1553).png';
import model2 from './Screenshot (1555).png';
import cover from './coverp.png';
import 'bootstrap/dist/css/bootstrap.min.css';

class App extends Component {

  state = {
    image: null,
    image2: null,
    processedImageData: null,
    processedImageData2: null,
  };

  

  handleImageChange = (e) => {
    this.setState({
      image: e.target.files[0],
      image2: e.target.files[0]
    });
  };

  handleSubmit = (e) => {
    e.preventDefault();
  
    if (this.state.image) {
      let form_data = new FormData();
      form_data.append('image', this.state.image, this.state.image.name);
  
      axios.post('http://localhost:8000/vgg16_unet/', form_data, {
        headers: {
          'content-type': 'multipart/form-data',
        },
        responseType: 'arraybuffer', // Request binary data
      })
        .then((res) => {
          // Convert the binary data to a base64 string
          const base64Data = btoa(
            new Uint8Array(res.data).reduce(
              (data, byte) => data + String.fromCharCode(byte),
              ''
            )
          );
          const processedImageData = `data:image/png;base64,${base64Data}`;
          this.setState({ processedImageData });
        })
        .catch((err) => console.log(err));
    } else {
      console.log('No image selected');
    }}

    handleSubmit2 = (e) => {
      e.preventDefault();
    
      if (this.state.image2) {
        let form_data = new FormData();
        form_data.append('image', this.state.image2, this.state.image2.name);
    
        axios.post('http://localhost:8000/resnet50_unet/', form_data, {
          headers: {
            'content-type': 'multipart/form-data',
          },
          responseType: 'arraybuffer', // Request binary data
        })
          .then((res) => {
            // Convert the binary data to a base64 string
            const base64Data = btoa(
              new Uint8Array(res.data).reduce(
                (data, byte) => data + String.fromCharCode(byte),
                ''
              )
            );
            const processedImageData2 = `data:image/png;base64,${base64Data}`;
            this.setState({ processedImageData2 });
          })
          .catch((err) => console.log(err));
      } else {
        console.log('No image selected');
      }}

  render() {
    return (
      <div className="App">
        
        <div class="bg-primary bg-gradient 
                    w-100% p-3 text-light"> 
            <h1><center><b>Breast Cancer Segmentation Models</b></center></h1>
        </div> 
        
        <div class="media">
        <img src={cover} class="img-thumbnail" alt="20% height image" style={{width: "100%", height: "10%"}}/>
        
        <div class="card text-left">
        <div class="card" style={{width: "10%"}}>
</div>


  <div class="card-header"><h3><center><b>Breast Cancer Awareness</b></center></h3>

  </div>
  <div class="flex-container"><center>
  <div class="card-body" style={{width: "80%"}}>
    
    <p class="card-text">Breast cancer is one of the most common types of cancer affecting millions of women worldwide, and it can also affect men, albeit less commonly. Awareness is crucial because early detection often leads to more effective treatment and a higher chance of survival.
</p><p>Segmentation helps pathologists in making precise diagnoses by providing clear images of cancer cells, which is crucial for determining the stage of cancer. This process involves the identification and isolation of cancerous cells from non-cancerous cells and the surrounding tissue in images obtained from breast tissue biopsies. The goal is to accurately define the boundaries of the malignant cells to assess the progression and potential aggressiveness of the cancer.</p>
    </div></center>
  </div>

</div>
<div class="flex-container"><center>
  <div class="media-body" style={{width: "70%"}}>
    <h5 class="mt-0"><center><b>Project Overview</b></center></h5>
    <p>The objective of the model is to segment the tumors in breast cancer slide images. With the result of the model, we can notice the tumors more easily</p>
    <p class="mb-0">To achieve this objective, we utilized cancer images and segmentation label from HistomicsUI (kitware.com) in which  the dataset associated with the paper: Amgad M, Elfandy H, ..., Gutman DA, Cooper LAD. Structured crowdsourcing enables convolutional segmentation of histology images. Bioinformatics. 2019. doi: 10.1093/bioinformatics/btz083.</p>
  
  </div></center></div>
</div>

<div id="list-example" class="list-group">
 
  <a class="list-group-item list-group-item-action" href="#list-item-1" style={{backgroundColor: '#6ebeff'}}><center><b><h2>ResNet50 + Unet Model (Click to test)</h2></b></center></a><center><img src={model1} class="img-thumbnail" alt="20% height image" style={{width: "50%"}}/><p><h5>Example of the Resnet50 result</h5></p></center>
  <div data-bs-spy="scroll" data-bs-target="#list-example" data-bs-offset="0" class="scrollspy-example" tabindex="0">
  <center><p id="list-item-1">Please upload cell image to see tumor segmentation result below</p></center>
  </div>
  <div class="flex-container"><center> 
  <form onSubmit={this.handleSubmit2}>
          <p>
            <input
              type="file"
              id="image"
              accept="image/png, image/jpeg" class="btn btn-primary"
              onChange={this.handleImageChange}
              required
            />
          </p>
          <input type="submit" value="Upload Image" class="btn btn-primary"/><p> </p>
        </form>
        
        {this.state.processedImageData2 && (
          <img src={this.state.processedImageData2} alt="Processed Image" />)}</center></div>
      
  <a class="list-group-item list-group-item-action" href="#list-item-2" style={{backgroundColor: '#6ebeff'}}><center><b><h2>VGG16 + Unet Model (Click to test)</h2></b></center></a><center><img src={model2} class="img-thumbnail" alt="20% height image" style={{width: "50%"}}/><p><h5>Example of the VGG16 result</h5></p></center>
<div data-bs-spy="scroll" data-bs-target="#list-example" data-bs-offset="0" class="scrollspy-example" tabindex="0">
  <center><p id="list-item-2">Please upload cell image to see tumor segmentation result below</p></center>

</div>

<div class="flex-container"><center>
        <form onSubmit={this.handleSubmit}>
          <p>
            <input
              type="file"
              id="image"
              accept="image/png, image/jpeg" class="btn btn-primary"
              onChange={this.handleImageChange}
              required
            />
          </p>
          
          <input type="submit" value="Upload Image" class="btn btn-primary"/><p> </p>
          <br></br>
        </form>
        
        {this.state.processedImageData && (
          <img src={this.state.processedImageData} alt="Processed Image" />)}</center></div></div>
          
          
          <div class="container">
  <footer class="py-3 my-4">
    <ul class="nav justify-content-center border-bottom pb-3 mb-3">
      <li class="nav-item">Developers</li>
    </ul>
    <p class="text-center text-muted">👩‍💻 Parinda Pannoon & Yanika Dontong 👩‍💻</p>
    <p class="text-center text-muted">&copy; Buy us beer 🍻</p>
  </footer>
</div>
          
          
          </div>
          
    );
    
  }
  
}


export default App;