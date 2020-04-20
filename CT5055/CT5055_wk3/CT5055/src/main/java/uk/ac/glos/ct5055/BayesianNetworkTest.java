package uk.ac.glos.ct5055;

import org.encog.ml.bayesian.BayesianEvent;
import org.encog.ml.bayesian.BayesianNetwork;
import org.encog.ml.bayesian.EventType;
import org.encog.ml.bayesian.query.enumerate.EnumerationQuery;

public class BayesianNetworkTest {

    public void testEncog(){
        // build the bayesian network structure
        BayesianNetwork network = new BayesianNetwork();

        BayesianEvent rained = network.createEvent("rained");
        BayesianEvent evenTemperatures = network.createEvent("temperature");
        BayesianEvent gardenGrew = network.createEvent("gardenGrew");
        BayesianEvent plentyOfCarrots = network.createEvent("carrots");
        BayesianEvent plentyOfTomatoes = network.createEvent("Tomatoes");

        network.createDependency(rained, gardenGrew);
        network.createDependency(evenTemperatures, gardenGrew);
        network.createDependency(gardenGrew, plentyOfCarrots);
        network.createDependency(gardenGrew, plentyOfTomatoes);
        network.finalizeStructure();

        // build the truth tales. Args: prob, result, args
        rained.getTable().addLine(0.2, true);
        evenTemperatures.getTable().addLine(0.5, true);

        gardenGrew.getTable().addLine(0.9, true, true, true);
        gardenGrew.getTable().addLine(0.7, true, false, true);
        gardenGrew.getTable().addLine(0.5, true, true, false);
        gardenGrew.getTable().addLine(0.1, true, false, false);

        plentyOfCarrots.getTable().addLine(0.8, true, true);
        plentyOfCarrots.getTable().addLine(0.2, true, false);
        plentyOfTomatoes.getTable().addLine(0.6, true, true);
        plentyOfTomatoes.getTable().addLine(0.1, true, false);

        // validate the network
        network.validate();

        // display basic stats
        System.out.println(network.toString());
        System.out.println("Parameter count: " + network.calculateParameterCount());

        EnumerationQuery query = new EnumerationQuery(network);
        //SamplingQuery query = new SamplingQuery(network);
        query.defineEventType(rained, EventType.Evidence);
        query.defineEventType(evenTemperatures, EventType.Evidence);
        query.defineEventType(plentyOfCarrots, EventType.Outcome);
        query.setEventValue(rained, true);
        query.setEventValue(evenTemperatures, true);
        query.setEventValue(plentyOfCarrots, true);
        query.execute();
        System.out.println(query.toString());
    }
}
