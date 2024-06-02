import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
from packaging import version
import tensorboard as tb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Check TensorBoard version
major_ver, minor_ver, _ = version.parse(tb.__version__).release
assert major_ver >= 2 and minor_ver >= 3, "This notebook requires TensorBoard 2.3 or later."
print("TensorBoard version: ", tb.__version__)

# Specify the relative path to the TensorBoard event file
event_file_path = 'log/ppo/SimpleDoorKey/experiment010/events.out.tfevents.1717148382.LC-0929.40668.0'

# Load the event data
event_acc = EventAccumulator(event_file_path)
event_acc.Reload()

# Convert event data to a pandas DataFrame
scalars = event_acc.Scalars('scalars')
df = pd.DataFrame(scalars)

# Visualize the data
sns.set(style="darkgrid")
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='step', y='value', hue='tag')
plt.title('TensorBoard Scalars')
plt.xlabel('Step')
plt.ylabel('Value')
plt.show()

# Display the relative path
print(f"Data loaded from: {event_file_path}")
