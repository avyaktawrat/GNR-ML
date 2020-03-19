# Import necessary modules; CSV is for reading .csv files and Plots is for plotting
using CSV
using Plots
using DataFrames

# Y = β_0 * X_0 + β_1 * X_1 + β_2 * X_2
# if X_0 = 1: we can write the above as Y = β * X for vectorization
# Predictions from Model β: Y_Pred = β * X

# Read the dataset from file
dataset = CSV.read("C:\\Users\\Avyakta\\github\\GNR-ML\\Assignment1\\data\\housingPriceData.csv")

dataset = DataFrame(dataset)
function partitionTrainTest(data, at)
    n = nrow(data)
    #n = 1000
    idx = (1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    return data[train_idx,:], data[test_idx,:]
end
function standard(X)
    mean = sum(X)/length(X)
     var = sum((X .- (sum(X)/length(X))).^2) / (length(X))
     sd  = sqrt(var)
     Y = (X .- mean)/(sd)
    return Y
end
#using RDatasets
#iris = dataset("datasets", "iris")
train,test = partitionTrainTest(dataset, 0.8)

course1 = train.price
course2 = standard(train.bedrooms)
course3 = standard(train.sqft_living)
course4 = standard(course2.^2)
course5 = standard(course3.^2)
course6 = standard(course2.*course3)

#mean_sq_price = mean(course1)
#course1 = course1 .- mean_sq_price
#
# # Visualize the columns as a scatter plot
#display(scatter3d(course2, course3, course4))
#
# # Stub column 1 for vectorization.
m = length(course1)
x0 = ones(m)
#
# # Define the features array
X = cat(x0, course2, course3, course4, course5, course6, dims=2)
# column wise concatenation => dims=2
# Get the variable we want to regress
Y = course1

# Define a function to calculate cost function
function costFunction(X, Y, B)
    m = length(Y)
    cost = sum(((X * B) - Y).^2)/(2*m)
    return cost
end

# # Initial coefficients
B = zeros(6, 1)
# Calcuate the cost with intial model parameters B=[0,0,0]
intialCost = costFunction(X, Y, B)


# Define a function to perform gradient descent
function gradientDescent(X, Y, B, learningRate, numIterations)
    costHistory = zeros(numIterations)
    m = length(Y)
    # do gradient descent for require number of iterations
    for iteration in 1:numIterations
        # Predict with current model B and find loss
        loss = (X * B) - Y
        # Compute Gradients: Ref to Andrew Ng. course notes linked on course page and Moodle
        gradient = (X' * loss)/m
        # Perform a descent step in direction oposite to gradient; we want to minimize cost!
        B = B - learningRate * gradient
        # Calculate cost of the new model found by descending a step above
        cost = costFunction(X, Y, B)
        # Store costs in a vairable to visualize later
        costHistory[iteration] = cost
    end
    return B, costHistory
end

#
learningRate = 0.003
newB, costHistory = gradientDescent(X, Y, B, learningRate, 20000)

# Make predictions using the learned model; newB
course1_test = test.price
course2_test = standard(test.bedrooms)
course3_test = standard(test.sqft_living)
course4_test = standard(course2_test.^2)
course5_test = standard(course3_test.^2)
course6_test = standard(course2_test.*course3_test)

m_test = length(course1_test)
x0_test = ones(m_test)

X_test = cat(x0_test, course2_test, course3_test, course4_test,course5_test, course6_test, dims=2)

YPred = X_test * newB

# visualize and compare the the prediction with original; below we plot only first 10 entries; plot! is to plot on the existing plot window
Y_test = test.price
#display(plot(Y_test[1:10]))
#display(plot!(YPred[1:10]))

# Visualize the learning: how the loss decreased.
#display(plot(costHistory))

rmse = ((sum((YPred.-Y_test).^2))/m_test)^0.5

r_sq = 1 - (sum((YPred.-Y_test).^2))/(sum((Y_test.-(sum(Y_test)/length(Y_test))).^2))

df = DataFrame(YPred)
CSV.write("C:\\Users\\Avyakta\\github\\GNR-ML\\Assignment1\\data\\1b.csv", df)
