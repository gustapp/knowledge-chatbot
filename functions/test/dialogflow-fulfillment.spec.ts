import {} from "ts-jest";

import * as chai from "chai";
const expect = chai.expect;

import * as request from 'supertest';

const url = 'http://localhost:5000';

describe("POST /sofia_fb13_fullfilment - fulfillment function endpoint", ()=> {

    /**
     * @user  What is the religion of antoine_brutus_menier ? 
     * @chatbot antoine_brutus_menier's religion is catholicism
     */
    it("Predict Tail: Should return tail: `Catholicism`", () => {

        const tailPredictionReq = require('./mock/tail-prediction.json');

        return request(url).post('/sofia_fb13_fullfilment')
            .send(tailPredictionReq)
            .expect(200)
            .expect(response => {
                expect(response.text)
                    .to.be.equal(`{"fulfillmentText":"antoine_brutus_menier's religion is catholicism"}\n`);
        });
    });
    /**
     * @user  Tell me 2 profession monarchs  ? 
     * @chatbot monarch1, monarch2
     */
    it("Predict Head: Should return tail: `Monarchs`", () => {

        return false;
    });
    /**
     * @user  antoine_brutus_menier's religion is catholicism ? 
     * @chatbot Yes
     */
    it("Predict Triple: Should return veracity: `Yes`", () => {

        return false;
    });
});
