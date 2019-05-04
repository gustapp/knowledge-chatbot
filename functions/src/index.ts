import * as functions from 'firebase-functions';
import * as admin from 'firebase-admin';
import { WebhookClient } from 'dialogflow-fulfillment';
 
process.env.DEBUG = 'dialogflow:debug';

admin.initializeApp(functions.config().firebase);

/**
 * @function helloWorld
 * Test function
 */
export const helloWorld = functions.https.onRequest((request, response) => {
  response.send("Hello! Lets chat! lol");
});

/**
 * @function dialogflowFirebaseFulfillment
 * Dialogue fulfillment function for ganimedes chatbot
 */
exports.dialogflowFirebaseFulfillment = functions.https.onRequest((request, response) => {
  const agent = new WebhookClient({ request, response });
  console.log('Dialogflow Request headers: ' + JSON.stringify(request.headers));
  console.log('Dialogflow Request body: ' + JSON.stringify(request.body));
 
  function welcome(agent) {
    agent.add(`Welcome to my agent!`);
  };

  function fallback(agent) {
    agent.add(`I didn't understand`);
    agent.add(`I'm sorry, can you try again?`);
  };
  
  function religion(agent) {
  	agent.add(`Aye sir`);
  };

  // Run the proper function handler based on the matched Dialogflow intent name
  const intentMap = new Map();
  intentMap.set('Default Welcome Intent', welcome);
  intentMap.set('Default Fallback Intent', fallback);
  intentMap.set('Religion of Person Intent', religion);
  agent.handleRequest(intentMap);
});
