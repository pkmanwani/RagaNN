import numpy as np
from matplotlib import pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
from numpy import argmax, diff
from matplotlib.mlab import find
import scipy.stats as stat
from collections import Counter

def ac(sig):
#mirroring and then getting one of the correlated part
	corr = fftconvolve(sig, sig[::-1], mode='full')
	return corr[len(corr)//2:]

'''def leveler(data,n):

	for j in range(n):
		for i in range(1,len(data)-1,1):
			
			curr = data[i]
			prev = data[i-1]
			next = data[i+1]
			if(not next == 0 and not prev == 0):
				if(abs(curr / prev) > abs(next / curr)):
					data[i] = data[i+1]
				elif(abs(curr / prev) < abs(next / curr)):
					data[i] = data[i-1]
				else:
					data[i] = data[i-1]
			elif (prev == 0):
				data[i] = next
			else:
				data[i] = prev
				
	return data
'''
def noise_removal(vals):
	notes_out = vals
	for i in range(1,len(vals)-1):
		if((abs(vals[i-1] - vals[i]) > 10) and abs(vals[i+1] - vals[i])>10):
			print(i,vals[i])
			notes_out[i] = vals[i-1]
	return notes_out


def quantizer(vals, root):
	note_out = np.zeros(len(vals))
	notes_freqs = [root*(((1.0594**0.2)**item)) for item in range(-70,70)]
	#notes_freqs = np.concatenate(notes_freqs,[root*((1.0594**0.2)**item) for item in range(0,70)])
	for i in range(len(vals)):
		min_val = 100000
		for j in range(len(notes_freqs)-1):
			if(abs(notes_freqs[j]*(notes_freqs[j] - vals[i])) < min_val):
				min_val = abs(notes_freqs[j]*(notes_freqs[j] - vals[i]))
				note_out[i] = notes_freqs[j]
	return note_out,notes_freqs
			
	

samplerate, data = wavfile.read('sample1.wav')
scale = 64				

window_length = samplerate/scale

times = np.arange(len(data)/float(window_length))
length = len(data)
vals = np.zeros(len(times))

#plt.plot(data)
#plt.show()

#print times

#print samplerate
#print length
#print window_length

#print len(data)/float(samplerate)
#print len(times)
#print len(vals)
#print length-(length%samplerate)-window_length
#print (length-length%samplerate)-window_length

#print data
plt.ion()

i = 0
j = 0
for i in range(0, int(length-(length%samplerate)), int(window_length) ):
	#print i
	#print j
	current_data = data[i : i + int(window_length)]
	#current_data = np.reshape(current_data, len(current_data))
	#print current_data
	
	
	current_out = ac(current_data)

	d = diff(current_out)
	start = find(d > 0)[0]
	peak = argmax(current_out[start:]) + start

	if (samplerate/peak > 40 and samplerate/peak < 600 ):
		vals[j] = samplerate/peak
	else: vals[j] = 0
	print(vals[j])
	j = j+1	
	


	'''
	plt.clf()
	plt.plot(current_out)
	plt.show()
	plt.pause(0.000001)
	'''
plt.clf()


#vals = leveler(vals,10)
#plt.scatter(times,vals, marker = "o", linewidth = 0.005, color = 'r', vmin = 0, vmax = 600)

'''
z = np.polyfit(times, vals,50)
p = np.poly1d(z)
plt.plot(times, vals, '.', times, p(times), '-')
#plt.plot(times,vals,'b')
'''

plt.show()
#print np.mean(vals)
#print stat.mode(vals)



ctr = Counter(vals.ravel())
notes = [item[0] for item in ctr.most_common(60)]
print(notes)
quantized = quantizer(vals,notes[0])
out = noise_removal(quantized[0])

#print(quantized[1])
plt.scatter(times,out, marker = "o", linewidth = 0.005, color = 'y', vmin = 0, vmax = 600)
#plt.scatter(times,quantized[0], marker = ".", linewidth = 0.005, color = 'b', vmin = 0, vmax = 600)
'''
notes_in_words = freq_to_note(vals)
print notes_in_words

ctr = Counter(notes_in_words.ravel())
notes = [item[0] for item in ctr.most_common(12)]
print notes

thefile = open('test.txt', 'w')
for i in range(0,len(notes_in_words)):
	thefile.write("%c\n" %notes_in_words[i])
thefile.close()
'''

plt.pause(1000)



