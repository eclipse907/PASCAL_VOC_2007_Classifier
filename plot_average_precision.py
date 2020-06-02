import matplotlib.pyplot as plt
import classifier_voc2007_activelearning
import classifier_voc2007

if __name__ == '__main__':
    least_confidence_sampling_data = classifier_voc2007_activelearning.main(seed_size=200, train_batch_size=40,
                                                                            test_batch_size=10,
                                                                            num_of_data_to_add=60,
                                                                            sampling_parameter="1")
    plt.plot(least_confidence_sampling_data[0], least_confidence_sampling_data[1], "r-", label="least_confidence")
    margin_sampling_data = classifier_voc2007_activelearning.main(seed_size=200, train_batch_size=40,
                                                                  test_batch_size=10,
                                                                  num_of_data_to_add=60,
                                                                  sampling_parameter="2")
    plt.plot(margin_sampling_data[0], margin_sampling_data[1], "g-", label="margin_sampling")
    entropy_sampling_data = classifier_voc2007_activelearning.main(seed_size=200, train_batch_size=40,
                                                                   test_batch_size=10,
                                                                   num_of_data_to_add=60,
                                                                   sampling_parameter="3")
    plt.plot(entropy_sampling_data[0], entropy_sampling_data[1], "b-", label="entropy_sampling")
    num_of_data, random_sampling_average_precision = classifier_voc2007.main(num_of_data=200, batch_size_train=40,
                                                                             batch_size_test=10)
    plt.scatter(num_of_data, random_sampling_average_precision, color="k")
    plt.annotate("random_sampling", (num_of_data, random_sampling_average_precision), textcoords="offset points",
                 xytext=(0, 10), ha='center')
    num_of_data, random_sampling_average_precision = classifier_voc2007.main(num_of_data=400, batch_size_train=40,
                                                                             batch_size_test=10)
    plt.scatter(num_of_data, random_sampling_average_precision, color="k")
    plt.annotate("random_sampling", (num_of_data, random_sampling_average_precision), textcoords="offset points",
                 xytext=(0, 10), ha='center')
    num_of_data, random_sampling_average_precision = classifier_voc2007.main(num_of_data=800, batch_size_train=40,
                                                                             batch_size_test=10)
    plt.scatter(num_of_data, random_sampling_average_precision, color="k")
    plt.annotate("random_sampling", (num_of_data, random_sampling_average_precision), textcoords="offset points",
                 xytext=(0, 10), ha='center')
    plt.xlabel("Number of samples trained on")
    plt.ylabel("Average precision")
    plt.legend()
    plt.show()
