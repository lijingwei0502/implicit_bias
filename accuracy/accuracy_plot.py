import matplotlib.pyplot as plt
import numpy as np

start_epoch = 0
num_epochs = 200    
skip_plot = 10

# Read data from the file
with open('accuracy.txt', 'r') as file:
    lines = file.readlines()

# Assuming each line in the file corresponds to each list in order
train_loss_list = [float(num) for num in lines[0].split()]
test_loss_list = [float(num) for num in lines[1].split()]
train_accuracy_list = [float(num) for num in lines[2].split()]
test_accuracy_list = [float(num) for num in lines[3].split()]
average_region_list = [float(num) for num in lines[4].split()]
#variance_region_list = [float(num) for num in lines[5].split()]

plt.figure(figsize=(10, 8))
plt.plot(range(start_epoch, start_epoch + num_epochs + 1, skip_plot), average_region_list, label='Average Regions')
# plt.fill_between(range(start_epoch, start_epoch + num_epochs + 1, skip_plot), 
#                 np.array(average_region_list) - np.array(variance_region_list), 
#                 np.array(average_region_list) + np.array(variance_region_list), 
#                 alpha=0.5, label='Variance')
plt.title('Average number of regions over Epochs', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Average number of regions', fontsize=18)
plt.legend(fontsize = 16)  # Now this will work because elements have labels
plt.savefig('average_region.png')
plt.close()

# plt.figure()
# plt.plot(range(start_epoch, start_epoch + num_epochs + 1, args.skip_plot), average_entropy_list, label='Average Entropy')
# plt.fill_between(range(start_epoch, start_epoch + num_epochs + 1, args.skip_plot),
#                 np.array(average_entropy_list) - np.array(variance_entropy_list),
#                 np.array(average_entropy_list) + np.array(variance_entropy_list),
#                 alpha=0.5, label='Variance')
# plt.title('Average Entropy over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Average Entropy')
# plt.legend()  # Now this will work because elements have labels
# plt.savefig(args.dir + '/average_entropy.png')
# plt.close()

plt.figure(figsize=(10, 8))
plt.plot(range(start_epoch, start_epoch + num_epochs + 1), train_loss_list, label='Train Loss')
plt.plot(range(start_epoch, start_epoch + num_epochs + 1), test_loss_list, label='Test Loss')
plt.title('Train and Test Loss Over Epochs', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend(fontsize = 16)
plt.savefig('loss_curve.png') 
plt.close()  

plt.figure(figsize=(10, 8))
plt.plot(range(start_epoch, start_epoch + num_epochs + 1), train_accuracy_list, label='Train Accuracy')
plt.plot(range(start_epoch, start_epoch + num_epochs + 1), test_accuracy_list, label='Test Accuracy')
plt.title('Train and Test Accuracy Over Epochs', fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=14)
plt.xlabel('Epochs', fontsize=18)
plt.ylabel('Accuracy (%)', fontsize=18)
plt.legend(fontsize = 16)
plt.savefig('accuracy_curve.png') 
plt.close()