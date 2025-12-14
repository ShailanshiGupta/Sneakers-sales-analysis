# =============================================================================
# SNEAKER RESALE VALUE PREDICTION PROJECT
# =============================================================================
# SECTION 1: SETUP - Installing and Loading Packages
# =============================================================================

# Instaling packages 
install.packages("tidyverse")
install.packages("lubridate")
install.packages("rpart")

# Loading libraries 
library(tidyverse)
library(lubridate)
library(readr)
library(rpart)

# =============================================================================
# SECTION 2: DATA LOADING
# =============================================================================

# Seting working directory to Desktop
setwd("~/Desktop")

# Loading sneaker data
sneakers <- read.csv(file.choose())

# =============================================================================
# SECTION 3: INITIAL DATA EXPLORATION
# =============================================================================

# Viewing first few rows
head(sneakers)
# Checking dimensions (rows x columns)
dim(sneakers)
# Checking column names
colnames(sneakers)
# Summary statistics
summary(sneakers)

# =============================================================================
# SECTION 4: DATA CLEANING
# =============================================================================

# Checking price formats (they have $ and commas)
head(sneakers$Sale.Price)
head(sneakers$Retail.Price)

# Cleaned prices: removed $ and commas, converted to numbers
sneakers$Sale.Price <- parse_number(sneakers$Sale.Price)
sneakers$Retail.Price <- parse_number(sneakers$Retail.Price)

# Verifying cleaning worked
summary(sneakers$Sale.Price)
summary(sneakers$Retail.Price)

# Converting dates from text to date format
sneakers$Order.Date <- mdy(sneakers$Order.Date)
sneakers$Release.Date <- mdy(sneakers$Release.Date)

# Verifying date conversion
head(sneakers$Order.Date)
head(sneakers$Release.Date)

# =============================================================================
# SECTION 5: FEATURE ENGINEERING (Creating New Variables)
# =============================================================================

# Variable 1: Days Since Release (age of sneaker when sold)
sneakers$Days_Since_Release <- as.numeric(sneakers$Order.Date - sneakers$Release.Date)

# Variable 2: Markup in Dollars (profit amount)
sneakers$Markup_Dollar <- sneakers$Sale.Price - sneakers$Retail.Price

# Variable 3: Markup Percentage (profit %)
sneakers$Markup_Pct <- (sneakers$Markup_Dollar / sneakers$Retail.Price) * 100

# Variable 4: Log Transformations (for better regression)
sneakers$Log_Sale_Price <- log(sneakers$Sale.Price)
sneakers$Log_Retail_Price <- log(sneakers$Retail.Price)

# Variable 5: Is Premium Brand? (1=Yes, 0=No)
sneakers$Is_Premium <- ifelse(sneakers$Brand %in% c("Yeezy", "Off-White", "Jordan"), 1, 0)

# Variable 6: Extracting Year from Dates
sneakers$Release_Year <- year(sneakers$Release.Date)
sneakers$Sale_Year <- year(sneakers$Order.Date)

# Variable 7: Extracting Month from Dates
sneakers$Sale_Month <- month(sneakers$Order.Date)
sneakers$Release_Month <- month(sneakers$Release.Date)

# Variable 8: Did it Sell Above Retail? (1=Yes, 0=No)
sneakers$Above_Retail <- ifelse(sneakers$Sale.Price > sneakers$Retail.Price, 1, 0)

# Variable 9: Price Category (Budget/Mid-Range/Premium)
sneakers$Price_Category <- ifelse(sneakers$Retail.Price < 150, "Budget",
                                  ifelse(sneakers$Retail.Price < 220, "Mid-Range", "Premium"))

# Checking all new variables created
colnames(sneakers)

# Saving cleaned dataset
write.csv(sneakers, "sneakers_with_features.csv", row.names=FALSE)

# =============================================================================
# SECTION 6: MODEL BUILDING
# =============================================================================

# -----------------------------------------------------------------------------
# Model 1: Simple Linear Regression (Baseline)
# -----------------------------------------------------------------------------
model1 <- lm(Sale.Price ~ Retail.Price + Brand + Days_Since_Release, 
             data=sneakers)
summary(model1)

# Calculating performance
pred1 <- predict(model1, sneakers)
rmse1 <- sqrt(mean((sneakers$Sale.Price - pred1)^2))
r2_1 <- summary(model1)$r.squared

# -----------------------------------------------------------------------------
# Model 2: Multiple Linear Regression (Added Variables)
# -----------------------------------------------------------------------------
model2 <- lm(Sale.Price ~ Retail.Price + Brand + Days_Since_Release + 
               Is_Premium + Sale_Month + Release_Year + Buyer.Region, 
             data=sneakers)
summary(model2)

# Calculating performance
pred2 <- predict(model2, sneakers)
rmse2 <- sqrt(mean((sneakers$Sale.Price - pred2)^2))
r2_2 <- summary(model2)$r.squared

# -----------------------------------------------------------------------------
# Model 3: Log-Transformed Linear Regression
# -----------------------------------------------------------------------------
model3 <- lm(Log_Sale_Price ~ Log_Retail_Price + Brand + Days_Since_Release + 
               Sale_Month + Release_Year, 
             data=sneakers)
summary(model3)

# Calculating performance
pred3 <- exp(predict(model3, sneakers))  # Convert back from log
rmse3 <- sqrt(mean((sneakers$Sale.Price - pred3)^2))
r2_3 <- summary(model3)$r.squared

# -----------------------------------------------------------------------------
# Model 4: GLM (Generalized Linear Model)
# -----------------------------------------------------------------------------
glm_model <- glm(Sale.Price ~ Retail.Price + Brand + Days_Since_Release + 
                   Sale_Month + Release_Year, 
                 data = sneakers,
                 family = gaussian())
summary(glm_model)

# Calculating performance
glm_predictions <- predict(glm_model, sneakers)
glm_rmse <- sqrt(mean((sneakers$Sale.Price - glm_predictions)^2))
glm_r2 <- cor(sneakers$Sale.Price, glm_predictions)^2

# -----------------------------------------------------------------------------
# Model 5: Decision Tree
# -----------------------------------------------------------------------------
tree_model <- rpart(Sale.Price ~ Retail.Price + Brand + Days_Since_Release + 
                      Sale_Month + Release_Year,
                    data = sneakers)

# Calculating performance
tree_predictions <- predict(tree_model, sneakers)
tree_rmse <- sqrt(mean((sneakers$Sale.Price - tree_predictions)^2))
tree_r2 <- cor(sneakers$Sale.Price, tree_predictions)^2

# =============================================================================
# SECTION 7: MODEL COMPARISON
# =============================================================================

# Create comprehensive comparison table
comparison_final <- data.frame(
  Model = c("Linear Regression (Simple)", 
            "Linear Regression (Full)", 
            "Log-Transformed Regression",
            "GLM",
            "Decision Tree"),
  Technique = c("Linear", "Linear", "Linear", "Generalized Linear", "Tree-based"),
  R_Squared = c(r2_1, r2_2, r2_3, glm_r2, tree_r2),
  RMSE = c(rmse1, rmse2, rmse3, glm_rmse, tree_rmse)
)

# Display results
print(comparison_final)

# Identify best model
cat("\n🏆 BEST MODEL: Decision Tree\n")
cat("R-squared:", round(tree_r2, 4), "\n")
cat("RMSE: $", round(tree_rmse, 2), "\n")

# Save comparison
write.csv(comparison_final, "comparison_all_models.csv", row.names=FALSE)

# =============================================================================
# SECTION 8: VISUALIZATIONS
# =============================================================================

# Graph 1: Model Comparison (Side-by-side)
png("graph_all_models_comparison.png", width=1000, height=600)
par(mfrow=c(1,2))

barplot(comparison_final$R_Squared,
        names.arg = c("Linear\nSimple", "Linear\nFull", "Linear\nLog", "GLM", "Tree"),
        main = "R-Squared Comparison",
        ylab = "R-Squared",
        col = c("red", "skyblue", "lightgreen", "yellow", "gold"),
        ylim = c(0, 0.7))

barplot(comparison_final$RMSE,
        names.arg = c("Linear\nSimple", "Linear\nFull", "Linear\nLog", "GLM", "Tree"),
        main = "Error Comparison",
        ylab = "RMSE ($)",
        col = c("red", "skyblue", "lightgreen", "yellow", "gold"),
        ylim = c(0, 250))

dev.off()
par(mfrow=c(1,1))

cat("Model comparison graph saved!\n")

# Graph 2: Best Model - Actual vs Predicted
png("graph_tree_predictions.png", width=800, height=600)
plot(sneakers$Sale.Price, tree_predictions,
     xlab = "Actual Sale Price ($)",
     ylab = "Predicted Sale Price ($)",
     main = "Decision Tree: Actual vs Predicted",
     pch = 16,
     col = rgb(0, 0.5, 0, 0.3),
     xlim = c(0, 1000),
     ylim = c(0, 1000))
abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)
text(800, 200, paste("R² =", round(tree_r2, 3)), 
     cex = 1.5, col = "darkgreen")
dev.off()

cat("Decision Tree prediction graph saved!\n")

# =============================================================================
# SECTION 9: SAVE FINAL PROJECT
# =============================================================================

# Save all work
save.image("sneaker_project_FINAL.RData")

cat("\n✅ PROJECT COMPLETE!\n")
cat("📊 5 models built across 3 techniques\n")
cat("🏆 Best model: Decision Tree (58% R²)\n")
cat("📈 All graphs saved to Desktop\n")
cat("💾 Project saved as: sneaker_project_FINAL.RData\n")

# Create a nice-looking table
library(gridExtra)
library(grid)

# Your comparison data
comparison_table <- data.frame(
  Model = c("Linear (Simple)", "Linear (Full)", "Log Regression", "GLM", "Decision Tree"),
  R_Squared = c("31%", "43%", "53%", "42%", "58%"),
  RMSE = c("$213", "$194", "$196", "$194", "$165"),
  Rank = c("5th", "4th", "2nd", "3rd", "1st 🏆")
)

# Save as image
png("model_comparison_table.png", width=800, height=400, bg="white")

# Create table plot
grid.table(comparison_table, 
           rows=NULL,
           theme=ttheme_default(
             core=list(fg_params=list(fontsize=14, fontface="bold")),
             colhead=list(fg_params=list(fontsize=16, fontface="bold"),
                          bg_params=list(fill="lightgray")),
             rowhead=list(fg_params=list(fontsize=14))
           ))

dev.off()

cat("Table saved to Desktop as: model_comparison_table.png\n")


# Set working directory
setwd("~/Desktop")

# Graph 3: Brand Markup Bar Chart
png("graph_brand_markups.png", width=800, height=600)

# Calculate average markup by brand
brand_summary <- sneakers %>%
  group_by(Brand) %>%
  summarise(Avg_Markup_Pct = mean(Markup_Pct, na.rm=TRUE),
            Count = n()) %>%
  arrange(desc(Avg_Markup_Pct))

# Create bar chart
barplot(brand_summary$Avg_Markup_Pct,
        names.arg = brand_summary$Brand,
        main = "Average Markup % by Brand",
        ylab = "Markup Percentage (%)",
        xlab = "Brand",
        col = c("gold", "coral", "steelblue", "lightgreen", "lightyellow"),
        las = 2,
        ylim = c(0, 350))

# Add percentage labels on bars
text(x = 1:nrow(brand_summary), 
     y = brand_summary$Avg_Markup_Pct + 15,
     labels = paste0(round(brand_summary$Avg_Markup_Pct, 0), "%"),
     cex = 1)

dev.off()
cat("Graph 3 saved!\n")



# Graph 4: Price Category Comparison
png("graph_price_categories.png", width=800, height=600)

# Calculate by price category
category_summary <- sneakers %>%
  group_by(Price_Category) %>%
  summarise(Avg_Markup = mean(Markup_Pct, na.rm=TRUE),
            Count = n())

# Reorder for logical display
category_summary$Price_Category <- factor(category_summary$Price_Category,
                                          levels = c("Budget", "Mid-Range", "Premium"))

# Create bar chart
barplot(category_summary$Avg_Markup,
        names.arg = c("Budget\n(<$150)", "Sweet Spot\n($150-$220)", "Premium\n(>$220)"),
        main = "Markup % by Price Category",
        ylab = "Average Markup %",
        col = c("lightcoral", "gold", "lightblue"),
        ylim = c(0, 300))

# Add percentage labels
text(x = 1:3,
     y = category_summary$Avg_Markup + 15,
     labels = paste0(round(category_summary$Avg_Markup, 0), "%"),
     cex = 1.2,
     font = 2)

# Highlight the winner
text(2, 270, "HIGHEST ROI!", cex = 1.3, col = "darkgreen", font = 2)

dev.off()
cat("Graph 4 saved!\n")