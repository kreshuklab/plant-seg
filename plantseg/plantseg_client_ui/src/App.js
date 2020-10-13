import React from 'react';
import logo from './logo.svg';
import './App.css';

import Button from 'react-bootstrap/Button'
import Accordion from 'react-bootstrap/Accordion'
import Card from 'react-bootstrap/Card'

function WrittenInputBox(props) {
  /*
  Props: 
    name: param name
    value: current value in box
    inputType: string representing a JS type
    currentPath: string, needs to be passed to the handler
    changeHandler: change callback
  */
  return(
    <div>
      {props.name}
      <input type={props.inputType} 
        value={props.value} 
        onChange={event => props.changeHandler(props.currentPath, event.target.value)} />
    </div>
  )
}

function BooleanTickBox(props) {
  /*
  Props: 
    name: param name
    value: current value in box
    currentPath: list of strings
    changeHandler: change callback
  */
  return (
    <div>
      {props.name}
      <input type='checkbox' 
        onChange={event => props.changeHandler(props.currentPath, event.target.checked)}
        checked={props.value} />
    </div>
  )
}

function ArrayLayout(props) {
  /* 
  Display and modify an array of elements with no name,
  for example a list of files, or a list of numbers.

  This object might receive an array of objects, in which
  case, it will lay them out as a collection of TreeLayout-s.

  Props:
    name: box title
    valueArray: Array with elements to display
    currentPath: list of strings
    changeHandler: function
  */
  let list_of_elements = [];
  let el;

  for (let idx in props.valueArray) {
    el = props.valueArray[idx];

    if (typeof(el) === 'object') {
      let newPath = Array.from(props.currentPath);
      newPath.push(idx);

      if (Array.isArray(el)) {
        list_of_elements.push(<ArrayLayout 
          name="Unnamed Array" 
          valueArray={el} 
          currentPath={newPath}
          changeHandler={props.changeHandler} />)
      } else {
        list_of_elements.push(<TreeLayout 
          parameterDict={el} 
          currentPath={newPath} 
          changeHandler={props.changeHandler} />);
      }

    } else {
      // TODO Add event handler here
      list_of_elements.push(
        <div>
          <p>{el}</p>
        </div>
      )
    }
  }

  return (
    <div>
      <h3>{props.name}</h3>
      {list_of_elements}
    </div>
  )
}

function ParameterElement(props) {
  /*
  Props:
    paramName: string, name of the changeable parameter
    param: parameter to be returned, could be different types
    currentPath: list of strings
    changeHandler: function
  */
  let output;

  if ( ( typeof(props.param) === 'string' ) || (typeof(props.param) === 'number') ) {
    output = (
      <WrittenInputBox 
        name={props.paramName} 
        value={props.param} 
        inputType={typeof(props.param)} 
        currentPath={props.currentPath}
        changeHandler={props.changeHandler}/>
      )
  } else if ( typeof(props.param) === 'boolean' ) {
    output = <BooleanTickBox 
      name={props.paramName} 
      value={props.param} 
      currentPath={props.currentPath}
      changeHandler={props.changeHandler} />
  } else if ( props.param == null ){
    output = <h1>{props.paramName} null</h1>
  } else {
    output = <h1> Undefined Element, please fix me! </h1>
  }
      
  return output
}

function TreeLayout(props) {
  /*
  Props:
    parameterDict: object, dictionary of name-parameter pairs
    currentPath: list of strings. Shows the location of this element
      in the state layout tree.
    changeHandler: function, passed on to child elements
   */
  let output = [];

  let val; 
  for (let key in props.parameterDict) {

    if (props.parameterDict.hasOwnProperty(key)) {
      val = props.parameterDict[key];
      let newPath = Array.from(props.currentPath);
      newPath.push(key);

      if (typeof( val ) === 'object') {

        if (Array.isArray( val )) {
          output.push( 
            <ArrayFileSelector
              name={key} 
              valueArray={val} 
              currentPath={newPath} 
              changeHandler={props.changeHandler}/>)
        } else {
          output.push( <TreeLayout 
            parameterDict={val} 
            currentPath={newPath}
            changeHandler={props.changeHandler}/> )
        }

      } else {
        output.push( 
          <ParameterElement 
            paramName={key} 
            param={props.parameterDict[key]} 
            currentPath={newPath}
            changeHandler={props.changeHandler}/> )

      }

    }
  }

  return output;

}

function SelectableText(props) {
  /*
  props:
    :text: string, text to display
    :isSelected: bool
    :onClick: handler
  */
  let output;
  if (props.isSelected) {
    output = <p style='color:blue;'>{props.text}</p>
  } else {
    output = <p>{props.text}</p>
  }
  return(
    <div>
      {output}
    </div>
  )
}

class SelectorBox extends React.Component {

  constructor(props) {

    this.state = {
      fileTree : {},
      selectedPath : ""
    }
    
    fetch("./datafiles")
    .then(res => res.json())
    .then(res => this.setState({selectedPath : "./", fileTree: res}))
  }

  moveInto() {
    //TODO
  }

  render() {
    return (
      <div>          

      </div>
    )
  }

}

class ArrayFileSelector extends React.Component {

  constructor(props) {
    /*
    Props:
      :name: Name of the box.
      :valueArray: array of string filenames.
      :currentPath: Used for updating selection
      :changeHandler: Used to updating selection
    */

    super(props);

    this.state = {
      selectorMode : false,
      collapsed : true,
      collapsedSize : 3
    }
  }

  switchCollapse() {
    let newState = Object.assign({}, this.state);
    newState.collapsed = !this.state.collapsed;
    this.setState(newState);
  }

  render() {
      let output;

      if (this.state.selectorMode) {

      } else {
        if (this.props.valueArray.length <= this.state.collapsedSize) {
          output = 
          <div>
            <ArrayLayout 
            name={this.props.name}
            valueArray={this.props.valueArray} 
            currentPath={this.props.currentPath} 
            changeHandler={this.props.changeHandler}/>
          </div>
        } else if (this.state.collapsed) {
          output = 
          <div>
            <div>
              <ArrayLayout 
              name={this.props.name}
              valueArray={this.props.valueArray.slice(0,this.state.collapsedSize)} 
              currentPath={this.props.currentPath} 
              changeHandler={this.props.changeHandler}/>
            </div>
            <div>
              <button onClick={() => this.switchCollapse()}>
                Expand ({this.props.valueArray.length - this.state.collapsedSize})
              </button>
            </div>
          </div>
        } else {
          output = 
          <div>
            <div>
              <ArrayLayout 
              name={this.props.name}
              valueArray={this.props.valueArray} 
              currentPath={this.props.currentPath} 
              changeHandler={this.props.changeHandler}/>
            </div>
            <div>
              <button onClick={() => this.switchCollapse()}>
                Contract
              </button>
            </div>
          </div>
        }
      }

      return (
        <div>
          {output}
          <button>Select</button>
        </div>
        )
  }

}

//class CollapsibleTile extends React.Component {
//  constructor(props) {
//    /*
//    Props:
//      :content: content to display when tile is expanded
//      :title: 
//    */
//    super(props);
//
//    this.state = {
//      expanded: false
//    }
//  }
//
//  render() {
//    let output;
//    if (this.state.expanded) {        
//      output = (
//        <div>
//          <div><button onClick={() => this.setState({expanded: false})}>{this.props.title}</button></div>
//          <div>{this.props.content}</div>
//        </div>)
//    } else {        
//      output = <button onClick={() => this.setState({expanded: true})}>{this.props.title}</button>
//    }
//
//    return output;
//  }
//}

class TaskCreationForm extends React.Component {
  
  constructor(props) {
    super(props)

    this.state = {
      available_layouts : {},
      selectedLayout : null,
      isSaving: false
    }

    this.eventKeysDict = {};
    this.eventKeysArray = [];

    this.initializeForm();

    this.modifyLayoutStateByPath = this.modifyLayoutStateByPath.bind(this);

    // Maybe add state info with all parameters?
    // TODO How to load parameter layouts? (number of params, names, etc)
  }

  getNewEventKey() {
    let newKey = this.eventKeysArray.length.toString();
    this.eventKeysArray.push(newKey);

    return newKey;
  }

  initializeKeysDict() {
    for (let branchName in this.state.available_layouts) {        
      let currentKeys = [];
      for (let subBranch in this.state.available_layouts[branchName]) {          
        currentKeys.push(this.getNewEventKey())
      }
      this.eventKeysDict[branchName] = currentKeys; 
    }
  }

  initializeForm() {
    fetch("/template_configs")
      .then(res => res.json())
      .then(res => (this.modifyState('available_layouts', res)))
      .then(() => this.initializeKeysDict())
  }

  modifyLayoutStateByPath(path, newVal) {
    /*
    This function is used to modify the leaf parameters of the 
    layout tree.

    Args:
      :param path: list of strings, used to index the tree.
      :param newVal: the new value. Type is checked just in case.
    */
    let availableLayoutsNew = (
      Object.assign({}, this.state.available_layouts)
    )
    let layoutCopy = (
      Object.assign({}, this.state.available_layouts[this.state.selectedLayout])
    );
    let movingRef = layoutCopy; // Points to a container, not value.

    for (let idx = 0; idx < path.length - 1; idx++) {
      let currentElement = path[idx];
      movingRef = movingRef[currentElement];
    }

    movingRef[ path[path.length - 1] ] = newVal;
    availableLayoutsNew[this.state.selectedLayout] = layoutCopy

    this.modifyState('available_layouts', availableLayoutsNew);
  }

  modifyState(paramName, newVal) {
    let newState = Object.assign({}, this.state); // shallow copy
    newState[paramName] = newVal;
    this.setState(newState);
  }

  handleChange(paramName, event) {
    let newVal = event.target.value;
    this.modifyState(paramName, newVal);
  }

  renderAccordion() {
      let outputArray = [];

      for (const [idx, branchName] of Object.keys(this.state.available_layouts[this.state.selectedLayout]).entries()) {          

        let currentEventKey = this.eventKeysDict[this.state.selectedLayout][idx];

        outputArray.push(
          <Accordion>
            <Card>
              <Accordion.Toggle as={Card.Header} eventKey={currentEventKey}>
                {branchName}
              </Accordion.Toggle>
              <Accordion.Collapse eventKey={currentEventKey}>
                <Card.Body>
                  <TreeLayout 
                    parameterDict={this.state.available_layouts[this.state.selectedLayout][branchName]} 
                    currentPath={[branchName]}
                    changeHandler={this.modifyLayoutStateByPath}/>
                </Card.Body>
              </Accordion.Collapse>
            </Card>
          </Accordion>
        )
      }

      return outputArray;
  }

  render() {

    var layoutSelectButtons;
    if (this.state.available_layouts === {}) {
      layoutSelectButtons = (
        <p>No layouts found...</p>
      )
    } else {
      layoutSelectButtons = 
        <div>
        {Object.keys(this.state.available_layouts).map( 
          key => <LayoutButton onClickHandler={name => this.modifyState('selectedLayout', name)} buttonDisplayName={key}/>)}
        </div>
    }

    var layoutBody;
    if (this.state.selectedLayout != null) {
        
      layoutBody = this.renderAccordion();

    } else {
      layoutBody = (
        <p>No layout selected</p>
      )
    }

    return (
      <div>
        <div>
          {layoutSelectButtons}
        </div>
        <div>
          {layoutBody}
        </div>
      </div>
    );
  } 
}

function LayoutButton(props) {
  return <button onClick={() => props.onClickHandler(props.buttonDisplayName)}>
    {props.buttonDisplayName}</button>
}

function StripButton(props) {
  return <Button 
    variant="primary"
    size="lg"
    onClick={() => props.onClickHandler(props.buttonId)}>
    {props.buttonDisplayName}</Button> 
}

function ConfigsElement(props) {
  return (
    <div>
      <span>{props.taskId}</span>
      <span>{props.modelName}</span> 
    </div>
  );
}

class ConfigsList extends React.Component {

  constructor(props) {
    super(props);

    this.state = {
      isLoading: true,
      currentResponse: [],
    }

    fetch("/tasks")
    .then(res => res.json())
    .then((res) => this.setState({isLoading: false, currentResponse: res.result}))
  }

  render() {

    let message = this.state.isLoading ? "Loading..." : "Configurations List";

    return (
        <div>
          <div>
            {message}
          </div>
          <div>
            {this.state.currentResponse.map((taskId) => <li>{taskId}</li>)}
          </div>
        </div>
      
      );

  }
}

class PageLayout extends React.Component {

  constructor(props) {
    super(props)

    // Layout states:
    // 0: Main page
    this.state = {
      mainLayoutState: 0
    }
  }

  stateChangeHandler(newState) {
    this.setState({mainLayoutState: newState})  
  }

  renderMainArea(mainState) {
    let pageOutput;
    switch (mainState) {
      case 0: 
        pageOutput = (
          <div>
            <ConfigsList />
          </div>
        );
        break;

      case 1:
        // Task creation parameters
        pageOutput = <TaskCreationForm />
        break;

      default:
        console.log("Page state index does not exist!")
    }

    return pageOutput;
  }

  renderButtonStrip() {
    return (
      <div className='buttonStrip'>
        <StripButton buttonId={0} onClickHandler={num => this.stateChangeHandler(num)} buttonDisplayName='Overview'/>
        <StripButton buttonId={1} onClickHandler={num => this.stateChangeHandler(num)} buttonDisplayName='Create Task'/>
      </div>
      );
  }

  render() {
    let mainArea = this.renderMainArea(this.state.mainLayoutState);
    let buttonStrip = this.renderButtonStrip();

    return (
      <div>
        <div>{buttonStrip}</div>
        <div>{mainArea}</div>
      </div>
    );
  }

}

function App() {
  return <PageLayout />
}

export default App;