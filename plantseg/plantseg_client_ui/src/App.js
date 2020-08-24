import React from 'react';
import logo from './logo.svg';
import './App.css';

// class PlantsegUI extends React.Component {
//   render() {
//     //TODO 
//   }
// }

class ConfigsList extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      isLoading: true,
      configFileNames: null,
    }

    console.log("fetching starts...")
    fetch("/tasks")
    .then(res => res.json())
    .then((res) => this.setState({isLoading: false, configFileNames: res}))
    .then(console.log("Fetching finished!"));
  }

  render() {

    let message = this.state.isLoading ? "Loading..." : "Under construction!";

    return <div>{message}</div>;
  }
}

function App() {
  console.log("App serve starting..")
  return (
    <ConfigsList />
  );
}

export default App;
