package cse417;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import java.util.function.DoubleFunction;

import static cse417.TeamModel.TEAMS;


/**
 * Program that finds {@code TeamModel}s that best fit provided data. The
 * option {@code --train} shows the results of using different penalties when
 * fitting historical data. Without that option, this just finds the best model
 * using the last {@code WEEKS} weeks of data. The latter supports using either
 * a fixed penalty or finding the model with a fixed number of non-zero params.
 */
public class TeamModeler {

  /** Number of weeks to use for building models. */
  private static int WEEKS = 6;

  /**
   * Tolerance parameter used by coordinate descent. The method will stop
   * iterating when the L0 norm of the change is less than this amount.
   */
  private static double TOLERANCE = 5e-3;

  /** Entry point for a program to build a model of NFL teams. */
  public static void main(String[] args) throws IOException {
    ArgParser argParser = new ArgParser("TeamModeler");
    argParser.addOption("train", Boolean.class);
    argParser.addOption("penalty", Double.class);
    argParser.addOption("num-nonzero", Integer.class);
    argParser.addOption("verbose", Boolean.class);
    args = argParser.parseArgs(args, 1, 1);

    if (argParser.hasOption("train")) {
      train(args[0], argParser.hasOption("verbose"));
    } else {
      // Find the drives in the last WEEKS worth of weeks.
      List<Drive> drives = loadDrives(args[0], 1, 16);
      int maxWeek = drives.stream().map((d) -> d.week)
          .max((a, b) -> Double.compare(a, b)).get();
      drives = drives.stream().filter((d) -> maxWeek - WEEKS + 1 <= d.week)
          .collect(Collectors.toList());

      TeamModel model;
      if (argParser.hasOption("num-nonzero")) {
        model = findBestSparseModel(drives,
            argParser.getIntegerOption("num-nonzero"), TOLERANCE,
            argParser.hasOption("verbose"));
      } else {
        double penalty = argParser.hasOption("penalty") ?
            argParser.getDoubleOption("penalty") : 0.0;
        model = findBestModel(drives, penalty, TOLERANCE,
            argParser.hasOption("verbose"));
      }
      model.printTo(System.out);
    }
  }

  /**
   * Returns the drives described in the given file from games in weeks of the
   * NFL season in the given range ({@code minWeek} to {@code maxWeek}).
   */
  private static List<Drive> loadDrives(
      String fileName, int minWeek, int maxWeek) throws IOException {
    List<Drive> drives = new ArrayList<Drive>();
    CsvParser parser = new CsvParser(fileName, true, new Object[] {
          String.class, String.class, Integer.class, Float.class, Float.class
        });
    while (parser.hasNext()) {
      String[] parts = parser.next();
      int week = Integer.parseInt(parts[2]);
      if (minWeek <= week && week <= maxWeek) {
        double expStartPts = Double.parseDouble(parts[3]);
        double expEndPts = Double.parseDouble(parts[4]);
        drives.add(new Drive(week, parts[0], parts[1], expStartPts, expEndPts));
      }
    }
    return drives;
  }

  /**
   * Fills in the given maps with the best model of how the offenses increase
   * the expected points in each drive and how the defenses decrease it.
   * @param drives List of the drives that the model should describe.
   * @param penalty Constant factor on the penalty term of the loss function.
   * @param tol Stop when the L0 change per iteration less than this amount.
   * @param verbose If true, prints progress of the model fitting process.
   */
  private static TeamModel findBestModel(
      final List<Drive> drives, double penalty, double tol, boolean verbose) {
    TeamModel best = new TeamModel();  // start with all zeros
    Boolean stop = false;
    while (!stop) {            
        TeamModel oldModel = best.copy();
        for (int k = 0; k < 65; k++) {
            final int i = k;
            DoubleFunction<Double> f = x -> best.evalLossWithX(drives, penalty, i, x); 
            double optimized = Optimizer.findMinimumOfUnimodal(f, -8, 8);  // call ternary search 
            if (i < 32) {
                best.setOffense(TEAMS.get(i), optimized);
            } else if (i < 64) {
                best.setDefense(TEAMS.get(i - 32), optimized);
            } else {
                best.setConstant(optimized);
            }
        }
        double norm0Diff = oldModel.norm0Diff(best);  
        //double norm0Diff = Math.abs(oldModel.norm0() - best.norm0());
        //System.out.println("norm0Diff = " + norm0Diff);
        if (norm0Diff < tol) {
            stop = true;
            break;
        }
    }
    // TODO: implement coordinate descent
    return best;
  }

  /**
   * Like {@code findBestModel} above but taking the desired number of non-zero
   * parameters in the model instead of the penalty.
   * <p>
   * If increasing the penalty by less than 0.0001 changes the number of
   * non-zero parameters from less than the desired amount to more than the
   * desired amount, then this will just return the model with fewer parameters
   */
  private static TeamModel findBestSparseModel(
      final List<Drive> drives, int numNonZeros, double tol, boolean verbose) {
    // TODO: Extra credit: implement this by figuring out which penalty to use
    //       to get the desired number of non-zero entries
      TeamModel best = new TeamModel();
      for (double penalty = 0; penalty <= 0.05; penalty += 0.0001) {
          TeamModel model = findBestModel(drives, penalty, tol, verbose);         
          int newCount = model.countNonZeroParameters(0.005);
          if (newCount <= numNonZeros) {
              best = model.copy();
              break;
          }
      }
      return best;
  }

  /** 
   * Prints the results of using different penalties .001 to 0.050 on each
   * period of {@code WEEKS} weeks in the provided data. That is, for each
   * period of {@code WEEKS+1} consecutive weeks in the data, it trains a model
   * using the first {@code WEEKS} of data (for each penalty) and then shows
   * the results for predicting the next week.
   * <p>
   * <b>Important</b>: Errors are normalized by subtracting off the minimum
   * error for any of the penalties tried for those weeks. This means, for each
   * period of {@code WEEKS} weeks, at least one penalty gives error zero.
   * (This normalization is necessary because some weeks are harder to predict
   * than others, and we want the model that works best on an average week not
   * just the worst week.)
   */
  private static void train(String fileName, boolean verbose) throws IOException {
    // TODO: try every period of WEEKS weeks in a row
    // TODO: try every penalty from 0.05 down to 0 stepping by 0.001
    // TODO: for each combination, print the number of non-zero entries and the
    //       amount of extra error beyond the minimum achieved by any penalty
    //       over those weeks of the data
    //  - (some weeks are just harder to fit than others, so normalize this way)      
      TeamModel model = new TeamModel();
      List<Integer> countList = new ArrayList<>(); 
      List<Double> errorList = new ArrayList<>(); 
      List<Double> minErrorList = new ArrayList<>();
      
      for (int i = 1; i <= 10; i++) {
          double minError = Integer.MAX_VALUE;
          List<Drive> drives = loadDrives(fileName, i, i + 5);
          for (int j = 50; j >= 0; j--) {
              double penalty = (double) j / 1000.0;
              long startTime = System.currentTimeMillis(); 
              model = findBestModel(drives, penalty, TOLERANCE, verbose);
              long endTime = System.currentTimeMillis(); 
              System.out.println("penalty: " + penalty + " running time: " + (endTime - startTime) + " ms"); // print the running time of each model computed
              
              int countNonZeros = model.countNonZeroParameters(0.005);
              countList.add(countNonZeros);
              // load the week of i + 6 to test the model 
              List<Drive> drive = loadDrives(fileName, i + 6, i + 6);
              double testError = model.evalLoss(drive, 0.00);
              errorList.add(testError);
              if (testError <= minError) minError = testError;  // find the minimum error on each period 
          }
          minErrorList.add(minError);
      }
                 
      System.out.println("minError: " + Arrays.toString(minErrorList.toArray()));
      System.out.println("countList: " + Arrays.toString(countList.toArray()));
      System.out.println("errrorList: " +  Arrays.toString(errorList.toArray()));
    
      // print out each of the combinations
      for (int i = 0; i < countList.size(); i++) {
          System.out.printf("count: %2d error: %g\n", countList.get(i), errorList.get(i) - minErrorList.get(i/51));
      }
   }
   
}
