

agents = [LLM_Agent(model1), LLM_Agent(model2), LLM_Agent(model3), LLM_Agent(model4), LLM_Agent(model5)]
boosted_model = CustomAdaBoostLLM(agents)
boosted_model.fit(X_train, y_train)
predictions = boosted_model.predict(X_test)
