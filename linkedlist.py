class EventStackNode:
    def __init__(self, event):
        self.event = event
        self.next = None
        self.prev = None


class EventStack:
    def __init__(self):
        self.head = None
        self.tail = None

    def is_empty(self):
        return self.head is None

    def insert(self, event):
        node = EventStackNode(event)

        # If the list is empty, add the new node as head and tail
        if self.is_empty():
            self.head = self.tail = node
            return

        # If the new node has lower time stamp than the head, insert at the beginning
        if node.event.time < self.head.event.time:
            node.next = self.head
            self.head.prev = node
            self.head = node
            return

        # If the new node has higher time stamp than the tail, insert at the end
        if node.event.time >= self.tail.event.time:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node
            return

        # Traverse the list and insert in sorted order (based on time stamp)
        current = self.head
        while current and current.event.time <= node.event.time:
            current = current.next

        # Insert the new node before the current node
        node.next = current
        node.prev = current.prev

        if current.prev:
            current.prev.next = node
        current.prev = node

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from an empty priority queue")

        # Pop the head (lowest time stamp event)
        event = self.head.event
        self.head = self.head.next

        if self.head:
            self.head.prev = None
        else:
            self.tail = None

        return event
