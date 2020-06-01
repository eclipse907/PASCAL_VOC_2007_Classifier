import matplotlib.pyplot as plt
import classifier_voc2007_activelearning
import classifier_voc2007

if __name__ == '__main__':
    least_confidence_sampling_data = classifier_voc2007_activelearning.main("1")
    plt.plot(least_confidence_sampling_data[0], least_confidence_sampling_data[1], "r-", label="least_confidence")
    margin_sampling_data = classifier_voc2007_activelearning.main("2")
    plt.plot(margin_sampling_data[0], margin_sampling_data[1], "g-", label="margin_sampling")
    entropy_sampling_data = classifier_voc2007_activelearning.main("3")
    plt.plot(entropy_sampling_data[0], entropy_sampling_data[1], "b-", label="entropy_sampling")
    num_of_data, random_sampling_average_precision = classifier_voc2007.main()
    plt.scatter(num_of_data, random_sampling_average_precision, color="k")
    plt.annotate("random_sampling", (num_of_data, random_sampling_average_precision), textcoords="offset points",
                 xytext=(0, 10), ha='center')
    num_of_data, random_sampling_average_precision = classifier_voc2007.main()
    plt.scatter(num_of_data, random_sampling_average_precision, color="k")
    plt.annotate("random_sampling", (num_of_data, random_sampling_average_precision), textcoords="offset points",
                 xytext=(0, 10), ha='center')
    num_of_data, random_sampling_average_precision = classifier_voc2007.main()
    plt.scatter(num_of_data, random_sampling_average_precision, color="k")
    plt.annotate("random_sampling", (num_of_data, random_sampling_average_precision), textcoords="offset points",
                 xytext=(0, 10), ha='center')
    plt.xlabel("Number of samples trained on")
    plt.ylabel("Average precision")
    plt.legend()
    plt.show()
